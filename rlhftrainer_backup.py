@beartype
class RLHFTrainer(nn.Module):

    def __init__(
        self,
        actor_critic: ActorCritic,
        reward_model: RewardLM,
        reference_model: BaseLM,
        config
    ):
        super().__init__()

        # take care of prompts -> token ids

        # assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        # if exists(prompts_path):
        #     path = Path(prompts_path)
        #     prompts = path.read_text().split('\n')

        # if exists(prompts):
        #     assert len(prompts) > 0, 'no prompts'
        #     assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'
        #     prompt_token_ids = tokenizer(prompts)

        # self.pad_value = pad_value # token pad value
        # self.num_prompts = prompt_token_ids.shape[0]
        # self.register_buffer('prompt_token_ids', prompt_token_ids)

        # models

        self.actor_critic = actor_critic

        self.reward_model = reward_model.eval()
        self.retokenization = config.retokenization

        # train hyperparameters

        self.n_update_epoch = config.n_update_epoch
        self.minibatch_size = config.minibatch_size
        self.max_norm = config.max_norm

        self.kl_div_loss_weight = config.kl_div_loss_weight

        # optimizers

        self.actor_optim = torch.optim.AdamW(
            self.actor_critic.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay
        )
        self.critic_optim = torch.optim.AdamW(
            self.actor_critic.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay
        )

        # ppo hyperparams

        self.eps_clip = config.eps_clip
        self.value_clip = config.value_clip
        self.beta_s = config.beta_s

        # # prepare with accelerator

        # (
        #     self.actor_critic,
        #     self.reward_model,
        #     self.actor_optim,
        #     self.critic_optim
        # ) = self.accelerate.prepare(
        #     self.actor_critic,
        #     self.reward_model,
        #     self.actor_optim,
        #     self.critic_optim
        # )

    def save(self, filepath = './checkpoint.pt'):
        torch.save(self.actor_critic.state_dict(), filepath)

    def load(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.actor_critic.load_state_dict(state_dict)

    @property
    def device(self):
        return self.actor_critic.actor.device

    @torch.no_grad()
    def generate(
        self,
        max_seq_len,
        *args,
        prompt,
        num_samples = 4,  # sample 4 per prompt and select the one with highest reward
        **kwargs
    ):
        assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
        prompt = repeat(prompt, 'n -> b n', b = num_samples)

        # actor_critic = self.accelerate.unwrap_model(self.actor_critic)
        # reward_model = self.accelerate.unwrap_model(self.reward_model)

        self.actor_critic.eval()

        (
            actions,
            sequences,
            mask,
            prompt_mask,
            action_logits,
            _
        ) = self.actor_critic.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            return_values = False,
            **kwargs
        )

        rewards = self.reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask,
            sample = True
        )

        best_sequence_index = rewards.topk(1, dim = -1).indices

        best_sequence = sequences[best_sequence_index]
        best_sequence = rearrange(best_sequence, '1 ... -> ...')

        return best_sequence

    def learn(
        self,
        memories: Deque[Memory]
    ):
        # stack all data stored in the memories

        all_memories_stacked_and_padded = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))

        # prepare dataloader for policy phase training

        # dl = create_dataloader(all_memories_stacked_and_padded, self.minibatch_size, device = self.device)
        dataloader = DataLoader(
            ExperienceDataset(
                all_memories_stacked_and_padded,
                device = self.device
            ),
            batch_size = self.minibatch_size,
            shuffle = True
        )

        self.actor_critic.train()

        # PPO training

        for _ in range(self.n_update_epoch):
            for (
                sequences,
                prompt_masks,
                masks,
                old_action_probs,
                old_log_probs,
                rewards,
                old_values
            ) in dataloader:
                action_masks = ~prompt_masks.bool() & masks

                action_logits, values = self.actor_critic(
                    sequences,
                    mask = action_masks
                )

                action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
                action_len = old_log_probs.shape[-1]

                action_probs = action_logits.softmax(dim = -1)
                action_log_probs = log_prob(action_probs, sequences)
                action_log_probs = action_log_probs[:, -action_len:]

                # calculate entropies, taking into account which part of the sequence is actually an action

                entropies = masked_entropy(action_probs, mask = action_masks)

                # calculate kl div between old action probs and new ones, taking into account which part of the sequence is action or not

                kl_penalty = 0.

                if self.kl_div_loss_weight > 0:
                    kl_penalty = masked_kl_div(old_action_probs, action_probs, mask = action_masks) * self.kl_div_loss_weight

                # subtract the kl penalty from the rewards

                rewards = rewards - kl_penalty

                # handle non-pooled values

                normalize_kwargs = dict()

                if old_values.ndim == 2:
                    old_values, values = map(lambda t: shift(t, shift = 1, dim = -2), (old_values, values))

                    old_values = old_values[:, -action_len:]
                    values = values[:, -action_len:]
                    rewards = rearrange(rewards, 'b -> b 1')
                    normalize_kwargs = dict(dim = -1, mask = action_masks[:, -action_len:])

                if values.ndim < rewards.ndim:
                    values = rearrange(values, '... -> ... 1')

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = masked_normalize(rewards - old_values, **normalize_kwargs)

                if advantages.ndim == 1:
                    advantages = rearrange(advantages, 'b -> b 1')

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropies

                # combine losses
                policy_loss = policy_loss.mean()

                # update actor
                policy_loss.backward()

                print(f'policy_loss: {policy_loss.item():.3f}')

                if self.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.actor_parameters(), self.max_norm)

                self.actor_optim.step()
                self.actor_optim.zero_grad()

                # calculate value loss and update value network separate from policy network

                value_loss = clipped_value_loss(values, rewards.detach(), old_values, self.value_clip)
                value_loss = value_loss.mean()

                print(f'critic_loss: {value_loss.item():.3f}')

                value_loss.backward()

                if self.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.critic_parameters(), self.max_norm)

                self.critic_optim.step()
                self.critic_optim.zero_grad()

    def train(
        self,
        prompts,
        n_episode = 50000,
        n_timestep = 500,
        n_update_timestep = 5000,
        max_batch_size = 16,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.
    ):
        device = self.device
        time = 0
        memories = deque([])

        prompts_ids = [prompt['input_ids'].squeeze().to(device) for prompt in prompts]
        prompts_text = [prompt['prompt_text'] for prompt in prompts]

        # for eps in tqdm(range(num_episodes), desc = 'episodes'):
        for episode in range(n_episode):
            for timestep in range(n_timestep):
                time += 1

                # select a bunch of random states (prompts)
                # and get the action (sampled sequence from palm as well as the action probs)
                # also calculate the reward using reward model and store

                rand_prompt_index = randrange(0, len(prompts_ids))

                state = prompts_ids[rand_prompt_index]
                prompt_text = prompts_text[rand_prompt_index]

                # remove padding from state

                # state_mask = state != self.pad_value
                # state = state[state_mask]

                # get predicted sequence

                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    action_logits,
                    value
                ) = self.actor_critic.generate(
                    rearrange(state, 'n -> 1 n'),
                    # state = state,
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                    return_values = True
                )
                # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
                action_logits = shift(action_logits, shift = 1, dim = -2)
                action_prob = action_logits.softmax(dim = -1)

                action_len = actions.shape[-1]
                action_log_prob = log_prob(action_prob, sequence)
                action_log_prob = action_log_prob[:, -action_len:]

                actions = rearrange(actions, '1 ... -> ...')

                # get reward as given by supervised trained reward model
                sequence = torch.cat((state, actions), dim = 0)

                prompt_length = len(state)
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length

                sequence = rearrange(sequence, 'n -> 1 n')
                prompt_mask = rearrange(prompt_mask, 'n -> 1 n').long()
                mask = default(mask, lambda: torch.ones(sequence.shape, dtype = torch.bool, device = device))

                # TODO
                if self.retokenization:
                    _, response_text = self.actor_critic.actor.decode_single(
                        input_ids = sequence,
                        attention_mask = mask,
                        prompt_mask = prompt_mask
                    )
                    reward_sequence, reward_mask, reward_prompt_mask = self.reward_model.encode_single(
                        prompt_text = prompt_text,
                        response_text = response_text
                    )
                else:
                    reward_sequence = sequence
                    reward_mask = mask
                    reward_prompt_mask = prompt_mask

                reward = self.reward_model(
                    reward_sequence,
                    attention_mask = reward_mask,
                    token_type_ids = reward_prompt_mask,
                    sample = True
                )

                detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')

                # store memory for learning
                memories.append(Memory(*map(detach_to_cpu_, (
                    sequence,
                    prompt_mask,
                    mask,
                    action_prob,
                    action_log_prob,
                    reward,
                    value
                ))))

                # learn from the stored memories
                if time % n_update_timestep == 0:
                    self.learn(memories)
                    memories.clear()

        print('rlhf training complete')
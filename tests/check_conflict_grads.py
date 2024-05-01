import numpy as np
import os
import seaborn as sns

# def plot_line():
#     figure = sns.lineplot(
#         data = self.train_histsorys,
#         x = 'Timestep',
#         y = col_name
#     )
#     figure.get_figure().savefig(
#         os.path.join(save_dir, col_name),
#         dpi = 400
#     )
#     plt.close()
    
if __name__ == '__main__':

    dir_path = './output/Panacea_train 2024-05-02 02-41-23'
    diags = np.load(os.path.join(dir_path, 'diags.npy'))
    conflict_cos_sims = np.load(os.path.join(dir_path, 'conflict_cos_sims.npy'))
    grad_conflict_scores = np.load(os.path.join(dir_path, 'grad_conflict_scores.npy'))

    print(diags, conflict_cos_sims, grad_conflict_scores)
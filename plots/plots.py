import matplotlib.pyplot as plt
import constants

def plot_one_loss(losses, client, method="baseline", current_run_name=constants.RUN_NAME):
    # global_loss = losses['global']
    global_loss = losses

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.xticks(range(0, len(global_loss) + 1, 10))
    
    ax.plot(global_loss, color='black', linestyle='-', label='Global Loss')

    # Add a legend
    ax.legend()

    # Add a title and axis labels
    ax.set_title(f'Average Losses Client {client + 1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    plt.savefig(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/average_loss_{method}_client{client}.png')

    plt.close()

def plot_one_loss_one_fold(losses, client, fold):
    global_loss = losses['global']

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.xticks(range(0, len(global_loss) + 1, 10))
    
    ax.plot(global_loss, color='black', linestyle='-', label='Global Loss')

    # Add a legend
    ax.legend()

    # Add a title and axis labels
    ax.set_title(f'Loss Client {client + 1} Fold {fold + 1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    plt.savefig(f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/loss_client{client}_fold{fold}.png')

    plt.close()
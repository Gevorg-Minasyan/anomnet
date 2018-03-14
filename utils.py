import matplotlib.pyplot as plt
import numpy as np
import os


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)


def create_results_dir():
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/gan_random_results'):
        os.mkdir('results/gan_random_results')
    if not os.path.isdir('results/gan_fixed_results'):
        os.mkdir('results/gan_fixed_results')
    if not os.path.isdir('results/dcgan_random_results'):
        os.mkdir('results/dcgan_random_results')
    if not os.path.isdir('results/dcgan_fixed_results'):
        os.mkdir('results/dcgan_fixed_results')



def save_results(gan, num_epoch, path = 'result.png', isFix=False, dim=(5, 5), img_size=(28, 28), figsize=(5, 5)):
    test_images = gan.generate(isFix, dim[0]*dim[1])
    test_images = test_images.reshape(dim[0]*dim[1], img_size[0], img_size[1])
        
    fig = plt.figure(figsize=figsize)
    for i in range(test_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(test_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
   
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()
    

def save_train_history(history, path = 'train_history.png'):
    x = range(len(history['D_losses']))

    y1 = history['D_losses']
    y2 = history['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def plot_timeseries(ts, pred, xTitle='', yTitle='',
                       title='', filename='test.html'):
    cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
#    cf_output = df.iplot(kind='scatter', title=title, xTitle=xTitle,
#                         online=False, asFigure=True)
    


    trace = go.Scatter(
        name='Measurement',
        #x=df['Time'],
        y=pred,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)')

    
    true_line = go.Scatter(
        name='Actual',
        #x=df['Time'],
        y=ts,
        mode='lines',
        line=dict(color='rgb(180, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='lines' )
    
    # Trace order can be important
    # with continuous error bars
    data = [trace, true_line]
    
    layout = go.Layout(
        yaxis=dict(title=yTitle),
        title=title,
        showlegend = True)
    fig = go.Figure(data=data, layout=layout)
    
    py.offline.iplot(fig, filename=filename)
    return data

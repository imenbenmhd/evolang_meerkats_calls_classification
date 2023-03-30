import torch
import numpy as np
import os
import sys
sys.path.insert(1,"/idiap/project/evolang/meerkats_imen/evolang.meerkats.call_type_classification/workspace/")
print(os.getcwd())

from src.models.Palazcnn import PalazCNN
from src.models.lit import Lit
import matplotlib.pylab as plt

if __name__== "__main__":
    path_="/idiap/project/evolang/meerkats_imen/evolang.meerkats.call_type_classification/workspace/" 
    device=torch.device('cpu')
    model = Lit(PalazCNN(n_output=6),learning_rate=1e-3,num_classes=6)
    print(model)
    #checkpoint = torch.load(path_+"meerkats-subseg/r90zx4pc/checkpoints/epoch=99-step=8400.ckpt",map_location=torch.device('cpu')) # path of kernel of 120 samples
    new_model=model.load_from_checkpoint("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/workspace/Isabel_meerkat/39gnunep/checkpoints/epoch=99-step=38100.ckpt") # path of kernel of 40 samples
    #new_model = model.load_from_checkpoint(checkpoint_path=path_+"/meerkats-subseg/2zu7aj5s/checkpoints/epoch=99-step=8400.ckpt") # path of kernel of 120 samples
    #new_model.global_pool.register_forward_hook(get_features('flatten'))
    kernel=new_model.model.conv1.weight.detach() # take the weights after first layer
    kernel=kernel.squeeze()
    xf = np.fft.fftfreq(1024, 1 / 16000) # the frequencies
    xf=xf[:len(xf) // 2]
    sumfft=np.fft.fft(kernel[0].numpy(),1024)
    sumfft=abs(sumfft[:len(sumfft) // 2]) # fft of first filter
    for i in range(1,40):
        fftnp=np.fft.fft(kernel[i].numpy(),1024)
        fftnp = fftnp[:len(fftnp) // 2]
        sumfft= sumfft + abs(fftnp)
    plt.plot(xf,abs(sumfft)/(np.max(sumfft)))
    plt.xlim((0,8000))
    plt.xlabel("Hz")
    plt.ylabel("amplitude")
    #plt.show()
    plt.savefig("120kernel_size.png")

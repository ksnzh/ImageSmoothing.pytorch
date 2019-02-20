# ImageSmoothing

原始实现 https://github.com/fqnchina/ImageSmoothing

CUDA 代码基本取自原始仓库，只是用pytorch做了简单扩展。

在pytorch 0.4.1 和 ubuntu 14.04 通过测试。

## 测试
输入为 RGB 格式，输入值范围在[0, 255]，输出值范围也是[0, 255]。

stylization和detail_enhance都可以测试下。 

测试样例：
```python2
from  fcn_sooth import FCNSmooth


checkpoint_path = "checkpoint/stylization/30_net_smooth.pth"
net = FCNSmooth().to(torch.device('cuda:0'))
net.load_state_dict(torch.load(checkpoint_path))

imgPath = ""
input_image = cv2.imread(imgPath)
input_tensor = torch.from_numpy(np.transpose(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), (2,0,1)))
input_tensor = input_tensor.unsqueeze(0).to(torch.device('cuda:0'), dtype=torch.float)

with torch.no_grad():
    net.eval()
    output_tensor = net(input_tensor)

output_image = output_tensor.squeeze(0).cpu().numpy()
output_image = cv2.cvtColor(np.transpose(output_image, (1,2,0)), cv2.COLOR_RGB2BGR)
```
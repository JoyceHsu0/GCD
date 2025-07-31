import os, sys
import torch, torch.nn.functional as F
import monai.transforms as mt

from dat.unetcnx import UNETCNX_A1 as MODEL # change to model file and name
PTH = 'dat/unetcnx.pth'     # parameter checkpoint file
SIZE = 128                  # size of model input, assume x=y=z
STRIDE = 112                # equals to SIZE - overlap
SPACING = (0.7, 0.7, 1.0)   # input image resample spacing
PERMUTE = (1, 2, 0)         # permute axes to match VTK x,y,z


class gcd_core():
    def __init__(self):
        # dummy initial values
        self.cam = torch.zeros ([256,256,150], dtype=torch.float32)
        self.img0 = None
        self.img1_spacing = SPACING
        self.layers = {'layer1':1}
        self.file_name = ''

    def load_and_process_input(self, input_file):
        self.file_name = input_file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img0 = mt.LoadImage()(input_file).flip(0)
        self.img1 = mt.Spacing(mode='bilinear', pixdim=SPACING)(self.img0.unsqueeze(0))

        # zero pad image if too small
        W, D = SIZE+STRIDE, SIZE
        shape = list(self.img1.shape)
        slices = [slice(None), slice(None), slice(None), slice(None)]
        if shape[1] < W: x = (W-shape[1])//2; slices[1] = slice(x, x+shape[1]); shape[1] = W
        if shape[2] < W: x = (W-shape[2])//2; slices[2] = slice(x, x+shape[2]); shape[2] = W
        if shape[3] < D: x = (D-shape[3])//2; slices[3] = slice(x, x+shape[3]); shape[3] = D
        if any(s.start is not None for s in slices):
            img = torch.zeros(shape)
            img[tuple(slices)] = self.img1
            self.img1 = img
            print('info: image is zero padded')

        self.img1 = mt.ScaleIntensityRange(a_min=-42, a_max=423, b_min=0, b_max=1, clip=True)(self.img1)
        self.img1_spacing = (SPACING[PERMUTE[0]], SPACING[PERMUTE[1]], SPACING[PERMUTE[2]])
        img2 = self.img1.unsqueeze(0).to(device)
        img2.requires_grad_()

        pth = torch.load(PTH, map_location='cpu')
        for key in list(pth['state_dict'].keys()):
            if 'ds' in key: # training only
                pth['state_dict'].pop(key)
        model = MODEL().to(device)
        model.load_state_dict(pth['state_dict'])
        model.eval()

        print('info: computing, this may take a while ', end='', flush=True)
        x0,y0,z0 = list((torch.tensor(self.img1[0].shape) -
            torch.tensor([STRIDE+SIZE,STRIDE+SIZE,SIZE]))//2)
        self.patch = []
        for (x,y) in [(0,0), (0,STRIDE), (STRIDE,0), (STRIDE,STRIDE)]:
            logits = model(img2[..., x0+x : x0+x+SIZE, y0+y : y0+y+SIZE, z0 : z0+SIZE])
            index = torch.argmax(logits[0], dim=0)
            loss = (logits[0, 1] * (index == 1)).sum()
            loss.backward()
            self.patch.append({k: (v.detach() * v.grad.detach()).cpu() for k, v in model.layers.items()})
            print('.', end='', flush=True)
        print(' done')

        self.layers = {k: v.size(1) for k, v in model.layers.items()}
        del model, img2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_cam(self, layer=None, n1=0, n2=999, use_overlay=True):
        if not layer:
            layer = list(self.layers.keys())[0]
        n2 = min(n2, self.layers[layer])
        shape = list(self.img1[0].shape)
        cam = torch.zeros (shape, dtype=torch.float32)
        x0,y0,z0 = list((torch.tensor(self.img1[0].shape) -
            torch.tensor([STRIDE+SIZE,STRIDE+SIZE,SIZE]))//2)
        for n,(x,y) in enumerate([(0,0), (0,STRIDE), (STRIDE,0), (STRIDE,STRIDE)]):
            q = torch.sum(self.patch[n][layer][:,n1:n2,...], dim=1).unsqueeze(0)
            q = F.interpolate(q, size=(SIZE,SIZE,SIZE), mode='trilinear')
            p = SIZE - STRIDE # overlap
            if n==0 or n==1:
                for i in range(p): q[0,0,STRIDE+i,:,:] *= (p-i)/p
            if n==2 or n==3:
                for i in range(p): q[0,0,       i,:,:] *=    i /p
            if n==0 or n==2:
                for i in range(p): q[0,0,:,STRIDE+i,:] *= (p-i)/p
            if n==1 or n==3:
                for i in range(p): q[0,0,:,       i,:] *=    i /p
            cam[x0+x : x0+x+SIZE, y0+y : y0+y+SIZE, z0 : z0+SIZE] += q[0,0]

        cam = torch.maximum(cam, torch.tensor(0))
        cam -= torch.min(cam)
        m = torch.max(cam)
        print(f'{m.item():.3f}', end=' ', flush=True)
        if m > 0:
            cam /= m
        cam[cam>0.1] += 1

        if use_overlay:
            cam[cam>0.1] += 1
            self.cam = (cam * 400 + self.img1[0] * 300).permute(*PERMUTE)
        else:
            self.cam = (cam * 900).permute(*PERMUTE)


if __name__ == '__main__':
    infile  = sys.argv[1] if len(sys.argv)>1 else 'dat/demo.1.nii.gz'
    layer   = sys.argv[2] if len(sys.argv)>2 else 'dec4'
    outfile = sys.argv[3] if len(sys.argv)>3 else 'cam'
    PERMUTE = (0,1,2)

    core = gcd_core()
    core.load_and_process_input(infile)
    core.compute_cam(layer)
    cam = F.interpolate(core.cam.unsqueeze(0).unsqueeze(0), size=list(core.img0.shape), mode='trilinear')
    mt.SaveImage() (cam[0,0], core.img0.meta, outfile)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_rulecode(nrule):
    rulecode=bin(nrule)[2:]
    for i in range(8-len(rulecode)):
        rulecode='0'+rulecode
    rulecode=rulecode.replace('0','9').replace('1','0').replace('9','1')
    return rulecode

def run_rule(r,c,image,rulecode):
    if image[r-1,c-1]==0:
        if image[r-1,c]==0:
            if image[r-1,c+1]==0:
                image[r,c]=rulecode[0]*1
            else:
                image[r,c]=rulecode[1]*1
        else:
            if image[r-1,c+1]==0:
                image[r,c]=rulecode[2]*1
            else:
                image[r,c]=rulecode[3]*1
    else:
        if image[r-1,c]==0:
            if image[r-1,c+1]==0:
                image[r,c]=rulecode[4]*1
            else:
                image[r,c]=rulecode[5]*1
        else:
            if image[r-1,c+1]==0:
                image[r,c]=rulecode[6]*1
            else:
                image[r,c]=rulecode[7]*1

def generate_automata(nrule,niter, panoramic=False):
    image=np.ones((niter,niter))
    image[0,(niter-1)//2]=0    
    rulecode = get_rulecode(nrule)
    for nraw in range(1,niter):
        for ncol in range(1,niter-1):
            run_rule(nraw,ncol,image,rulecode)
    if panoramic:
        image = image[niter//2:, :]
    return(1-image)
            
def generate_tag(nrule):
    rule = np.array([
        [1,1,1, 0, 1,0,1, 0, 1,0,0, 0, 1,1,1, 0,0,],
        [1,0,1, 0, 1,0,1, 0, 1,0,0, 0, 1,0,1, 0,0,],
        [1,0,1, 0, 1,0,1, 0, 1,0,0, 0, 1,0,0, 0,0,],
        [1,0,1, 0, 1,0,1, 0, 1,0,0, 0, 1,1,0, 0,0,],
        [1,1,0, 0, 1,0,1, 0, 1,0,0, 0, 1,0,0, 0,0,],
        [1,0,1, 0, 1,0,1, 0, 1,0,1, 0, 1,0,1, 0,0,],
        [1,0,1, 0, 1,1,1, 0, 1,1,1, 0, 1,1,1, 0,0,],
        ])
    pixel_digits = {
    0: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,1,1], 
    ]),
    1: np.array([
        [1,1,0], 
        [0,1,0], 
        [0,1,0], 
        [0,1,0], 
        [0,1,0], 
        [0,1,0], 
        [1,1,1], 
    ]),
    2: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [0,0,1], 
        [1,1,1], 
        [1,0,0], 
        [1,1,1], 
    ]),
    3: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [0,0,1], 
        [0,1,1], 
        [0,0,1], 
        [1,1,1], 
    ]),
    4: np.array([
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,1,1], 
        [0,0,1], 
        [0,0,1], 
    ]),
    5: np.array([
        [1,1,1], 
        [1,0,0], 
        [1,0,0], 
        [1,0,0], 
        [1,1,1], 
        [0,0,1], 
        [1,1,1], 
    ]),
    6: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,0], 
        [1,1,1], 
        [1,0,1], 
        [1,1,1], 
    ]),
    7: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [0,0,1], 
        [0,1,1], 
        [0,1,0], 
        [0,1,0], 
    ]),
    8: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,1,1], 
        [1,0,1], 
        [1,1,1], 
    ]),
    9: np.array([
        [1,1,1], 
        [1,0,1], 
        [1,0,1], 
        [1,0,1], 
        [1,1,1], 
        [0,0,1], 
        [1,1,1], 
    ])}
    space = np.array([
        [0], 
        [0], 
        [0], 
        [0], 
        [0], 
        [0], 
        [0], 
    ])
    
    digits = [int(i) for i in str(nrule)]
    digit_tag = np.concatenate([np.concatenate([pixel_digits[x], space], axis=1) for x in digits], axis=1)[:,:-1]
    tag = np.concatenate([rule, digit_tag], axis=1)
    return tag

def upscale_image(array, n=2):
    """Upscale a 2D or 3D array."""
    if len(array.shape) == 2:
        rows, cols = array.shape
        upscaled = np.repeat(np.repeat(array, n, axis=1), n, axis=0)
    elif len(array.shape) == 3:
        rows, cols, channels = array.shape
        upscaled = np.repeat(np.repeat(array, n, axis=1), n, axis=0)
    else:
        raise ValueError("Input array should be either 2D or 3D.")
    return upscaled

def get_color(c):
    black = np.array([0,0,0])

    palette = c[0]
    def get_palette(p):
        if p=="b":
            return sns.color_palette("bright")
        if p=="c":
            return sns.color_palette("colorblind")
        if p=="d":
            return sns.color_palette("dark")
        if p=="m":
            return sns.color_palette("muted")
        if p=="p":
            return sns.color_palette("pastel0")

def generate_explanation(nrule):
    rulecode = get_rulecode(nrule)
    sample_imgs=[]
    for x_0 in [0,1]:
        for x_1 in [0,1]:
            for x_2 in [0,1]:
                sample_imgs.append(np.array([[x_0,x_1,x_2], [0,0,0]]))

    combined_img = np.ones((2, 3*8+7))
    for i, img in enumerate(sample_imgs):
        run_rule(1,1,img, rulecode)
        combined_img[:,i*3+i:(i+1)*3+i]=img
    combined_img[1, ::2] = 1
    return 1-combined_img


def get_full_image(
        nrule = 30,
        niter = 1000, 
        space_ratio = 1/15, 
        expl_ratio = 3/4, 
        tag_ratio = 1/5, 
        panoramic = False, 
        ):

    tag = generate_tag(nrule)
    expl = generate_explanation(nrule)
    ca = generate_automata(nrule, niter, panoramic = panoramic)
    space = np.zeros((int(niter*space_ratio),niter))

    tag_length = niter*tag_ratio
    up_ratio = int(tag_length/tag.shape[1])
    tag_length = tag.shape[1]*up_ratio
    tag = upscale_image(tag,up_ratio)
    tag_x0 = int((niter-tag_length)//2)
    tag_x1 = tag_x0+tag.shape[1]
    tag_complete = np.zeros((tag.shape[0], niter))
    tag_complete[:, tag_x0:tag_x1] = tag
    
    expl_length = niter*expl_ratio
    up_ratio = int(expl_length/expl.shape[1])
    expl_length = expl.shape[1]*up_ratio
    expl = upscale_image(expl,up_ratio)
    expl_x0 = int((niter-expl_length)//2)
    expl_x1 = expl_x0+expl.shape[1]
    expl_complete = np.zeros((expl.shape[0], niter))
    expl_complete[:, expl_x0:expl_x1] = expl

    expl_y0 = ca.shape[0]+space.shape[0]
    expl_y1 = ca.shape[0]+space.shape[0]+expl_complete.shape[0]
    expl_rect = (expl_y0, expl_y1, expl_x0, expl_x1)

    img = np.concatenate([ca, space, expl_complete, space, tag_complete, space])
    
    return img, expl_rect

def color_image(img, background_color, foreground_color):
    final_img = np.zeros((img.shape[0], img.shape[1], 3))
    final_img[img == 0] = background_color
    final_img[img == 1] = foreground_color
    return final_img

def plot_clean(img, figsize=(20,20), upscale=1, expl_rect = None, color_rect = "black"):
    img = upscale_image(img, upscale)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')  # to hide the axes

    # Assuming the explanation is between expl_start and expl_end rows
    if expl_rect:
        step = upscale*(expl_rect[1]-expl_rect[0])//2
        for j, y in enumerate(range(expl_rect[0]*upscale, expl_rect[1]*upscale, step)):
            for i, x in enumerate(range(expl_rect[2]*upscale, expl_rect[3]*upscale, step)):
                if ((i+1)%4==0) or (i%2==0) and (j==1):
                    continue
                rect = plt.Rectangle((x-0.5, y-0.5), step, step, color=color_rect, linewidth=1.2, fill=False)
                ax.add_patch(rect)

    # Save the image without any white borders with dpi=100 to get 1000x1000 pixels for a 10x10 inch figure
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    return fig
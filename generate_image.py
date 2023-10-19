from utils import *
import os
from jsonargparse import CLI

def generate(
        nrule=30, 
        niter=200,
        space_ratio = 1/15, 
        expl_ratio = 3/4, 
        tag_ratio = 1/5, 
        outdir = "images", 
        upscale = 2, 
        color_rect = "white", 
):
    os.makedirs(outdir, exists_ok=True)
    img = get_full_image(nrule, niter, space_ratio, expl_ratio, tag_ratio)





if __name__=="__main__":
    CLI(generate)
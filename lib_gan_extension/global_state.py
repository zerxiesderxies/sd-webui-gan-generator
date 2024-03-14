
image_format: "png"
image_pad: 1.0

def init():
  global image_format
  global image_pad

def logger(*args):
    msg = " ".join(map(str, args))
    print(f"[GAN Generator] {msg}")

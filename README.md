# bayer2rgb

## Requirements

* opencv
* tensorflow
* keras
* numpy

## Usage

### Testing

    python reconstruct.py ./Original.bmp --output ./Reconstructed.bmp --mode=test --num_workers=1
    
### Processing
    python reconstruct.py ./RGB_CFA.bmp --output ./Reconstructed.bmp

### Details
    python reconstruct.py --help

    usage: reconstruct.py [-h] [--model MODEL] [--mode MODE] [--output OUTPUT]
                        [--gpu] [--num_workers NUM_WORKERS]
                        input

    positional arguments:
    input                 Input path. Should be color image

    optional arguments:
    -h, --help            show this help message and exit
    --model MODEL         Model path. Default is ./model.h5
    --mode MODE           Mode. Could be "test" or "process". Default is
                            "process"
    --output OUTPUT       Output path. Default is ./Reconstructed.bmp
    --gpu                 Flag for enabling gpu usage
    --num_workers NUM_WORKERS
                            Number of cpu cores used for computing
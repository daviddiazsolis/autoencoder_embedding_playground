import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784;
const NUM_IMAGES = 5000; // Load 5000 images for speed in browser
const FASHION_MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png';
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

export class MnistData {
  datasetImages: Float32Array | null = null;
  isFashion = false;

  async load() {
    let spritePath = FASHION_MNIST_IMAGES_SPRITE_PATH;
    
    // Check if Fashion MNIST is available, fallback to standard MNIST
    try {
      const res = await fetch(spritePath, { method: 'HEAD' });
      if (!res.ok) {
        spritePath = MNIST_IMAGES_SPRITE_PATH;
      } else {
        this.isFashion = true;
      }
    } catch (e) {
      spritePath = MNIST_IMAGES_SPRITE_PATH;
    }

    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (!ctx) throw new Error('Could not get canvas context');

    const imgRequest = new Promise<void>((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        canvas.width = 784; // 28x28 flattened
        canvas.height = NUM_IMAGES;
        ctx.drawImage(img, 0, 0, 784, NUM_IMAGES, 0, 0, 784, NUM_IMAGES);
        
        const imageData = ctx.getImageData(0, 0, 784, NUM_IMAGES);
        const datasetBytesView = new Float32Array(NUM_IMAGES * 784);
        
        for (let j = 0; j < imageData.data.length / 4; j++) {
          // All channels have the same value, so just read the red channel
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
        
        this.datasetImages = datasetBytesView;
        resolve();
      };
      img.onerror = reject;
      img.src = spritePath;
    });

    await imgRequest;
  }

  getTrainData(numExamples: number = 4000) {
    if (!this.datasetImages) throw new Error('Data not loaded');
    const xs = tf.tensor2d(
      this.datasetImages.slice(0, numExamples * IMAGE_SIZE),
      [numExamples, IMAGE_SIZE]
    );
    return { xs, ys: xs }; // Autoencoder: input is output
  }

  getTestData(numExamples: number = 1000) {
    if (!this.datasetImages) throw new Error('Data not loaded');
    const offset = 4000 * IMAGE_SIZE;
    const xs = tf.tensor2d(
      this.datasetImages.slice(offset, offset + numExamples * IMAGE_SIZE),
      [numExamples, IMAGE_SIZE]
    );
    return { xs, ys: xs };
  }
}

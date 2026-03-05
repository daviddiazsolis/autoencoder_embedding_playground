import * as tf from '@tensorflow/tfjs';

export class AutoencoderService {
  autoencoder: tf.LayersModel | null = null;
  encoder: tf.LayersModel | null = null;
  decoder: tf.LayersModel | null = null;

  buildModel(embeddingDim: number = 8) {
    // Instantiate layers
    const encoded1 = tf.layers.dense({ units: 256, activation: 'relu' });
    const encoded2 = tf.layers.dense({ units: 128, activation: 'relu' });
    const encoded3 = tf.layers.dense({ units: 64, activation: 'relu' });
    const embeddingLayer = tf.layers.dense({ units: embeddingDim, activation: 'linear', name: 'embedding' });

    const decoded1 = tf.layers.dense({ units: 64, activation: 'relu' });
    const decoded2 = tf.layers.dense({ units: 128, activation: 'relu' });
    const decoded3 = tf.layers.dense({ units: 256, activation: 'relu' });
    const outputLayer = tf.layers.dense({ units: 784, activation: 'sigmoid' });

    // Build Autoencoder
    const input = tf.input({ shape: [784] });
    const e1 = encoded1.apply(input);
    const e2 = encoded2.apply(e1);
    const e3 = encoded3.apply(e2);
    const emb = embeddingLayer.apply(e3);

    const d1 = decoded1.apply(emb);
    const d2 = decoded2.apply(d1);
    const d3 = decoded3.apply(d2);
    const out = outputLayer.apply(d3);

    this.autoencoder = tf.model({ inputs: input, outputs: out as tf.SymbolicTensor });
    this.autoencoder.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Build Encoder
    this.encoder = tf.model({ inputs: input, outputs: emb as tf.SymbolicTensor });

    // Build Decoder
    const decoderInput = tf.input({ shape: [embeddingDim] });
    const dec1 = decoded1.apply(decoderInput);
    const dec2 = decoded2.apply(dec1);
    const dec3 = decoded3.apply(dec2);
    const decOut = outputLayer.apply(dec3);

    this.decoder = tf.model({ inputs: decoderInput, outputs: decOut as tf.SymbolicTensor });
  }

  async evaluate(data: { xs: tf.Tensor2D; ys: tf.Tensor2D }) {
    if (!this.autoencoder) return null;
    const result = this.autoencoder.evaluate(data.xs, data.ys, { batchSize: 128 });
    const lossTensor = Array.isArray(result) ? result[0] : result as tf.Scalar;
    const lossArray = await lossTensor.data();
    if (Array.isArray(result)) result.forEach(t => t.dispose());
    else lossTensor.dispose();
    return lossArray[0];
  }

  async train(
    trainData: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    testData: { xs: tf.Tensor2D; ys: tf.Tensor2D },
    epochs: number,
    onEpochEnd: (epoch: number, logs: tf.Logs) => Promise<void> | void
  ) {
    if (!this.autoencoder) throw new Error('Model not built');

    await this.autoencoder.fit(trainData.xs, trainData.ys, {
      epochs: epochs,
      batchSize: 128,
      validationData: [testData.xs, testData.ys],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (logs) await onEpochEnd(epoch, logs);
          await tf.nextFrame(); // Yield to UI to show live updates
        }
      }
    });
  }

  getEmbedding(imageTensor: tf.Tensor2D): Float32Array {
    if (!this.encoder) throw new Error('Encoder not built');
    const embedding = this.encoder.predict(imageTensor) as tf.Tensor;
    const data = embedding.dataSync() as Float32Array;
    embedding.dispose();
    return data;
  }

  reconstructFromEmbedding(embeddingData: Float32Array | number[]): Float32Array {
    if (!this.decoder) throw new Error('Decoder not built');
    const embeddingTensor = tf.tensor2d(new Float32Array(embeddingData), [1, embeddingData.length]);
    const reconstructed = this.decoder.predict(embeddingTensor) as tf.Tensor;
    const data = reconstructed.dataSync() as Float32Array;
    embeddingTensor.dispose();
    reconstructed.dispose();
    return data;
  }
}

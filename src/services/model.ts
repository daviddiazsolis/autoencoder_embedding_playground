import * as tf from '@tensorflow/tfjs';

export type LayerConfig = {
  units: number;
  activation: string;
};

export class AutoencoderService {
  autoencoder: tf.LayersModel | null = null;
  encoder: tf.LayersModel | null = null;
  decoder: tf.LayersModel | null = null;

  buildModel(
    encoderLayers: LayerConfig[],
    decoderLayers: LayerConfig[],
    embeddingDim: number = 8,
    learningRate: number = 0.001,
    regularization: number = 0
  ) {
    const kernelRegularizer = regularization > 0 ? tf.regularizers.l2({ l2: regularization }) : undefined;

    // Build Encoder
    const input = tf.input({ shape: [784] });
    let lastLayer = input;

    const encoderDenseLayers: tf.layers.Layer[] = [];
    for (const config of encoderLayers) {
      const layer = tf.layers.dense({ 
        units: config.units, 
        activation: config.activation as any,
        kernelRegularizer 
      });
      encoderDenseLayers.push(layer);
      lastLayer = layer.apply(lastLayer) as tf.SymbolicTensor;
    }

    const embeddingLayer = tf.layers.dense({ 
      units: embeddingDim, 
      activation: 'linear', 
      name: 'embedding',
      kernelRegularizer
    });
    const emb = embeddingLayer.apply(lastLayer) as tf.SymbolicTensor;

    this.encoder = tf.model({ inputs: input, outputs: emb });

    // Build Decoder
    const decoderInput = tf.input({ shape: [embeddingDim] });
    let lastDecLayer = decoderInput;

    const decoderDenseLayers: tf.layers.Layer[] = [];
    for (const config of decoderLayers) {
      const layer = tf.layers.dense({ 
        units: config.units, 
        activation: config.activation as any,
        kernelRegularizer
      });
      decoderDenseLayers.push(layer);
      lastDecLayer = layer.apply(lastDecLayer) as tf.SymbolicTensor;
    }

    const outputLayer = tf.layers.dense({ 
      units: 784, 
      activation: 'sigmoid',
      kernelRegularizer
    });
    const decOut = outputLayer.apply(lastDecLayer) as tf.SymbolicTensor;

    this.decoder = tf.model({ inputs: decoderInput, outputs: decOut });

    // Build Full Autoencoder
    let aeLastLayer = input;
    for (const layer of encoderDenseLayers) {
      aeLastLayer = layer.apply(aeLastLayer) as tf.SymbolicTensor;
    }
    aeLastLayer = embeddingLayer.apply(aeLastLayer) as tf.SymbolicTensor;
    for (const layer of decoderDenseLayers) {
      aeLastLayer = layer.apply(aeLastLayer) as tf.SymbolicTensor;
    }
    const aeOut = outputLayer.apply(aeLastLayer) as tf.SymbolicTensor;

    this.autoencoder = tf.model({ inputs: input, outputs: aeOut });
    this.autoencoder.compile({ 
      optimizer: tf.train.adam(learningRate), 
      loss: 'meanSquaredError' 
    });
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

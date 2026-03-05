import React, { useEffect, useState, useRef } from 'react';
import { MnistData } from './services/mnist';
import { AutoencoderService } from './services/model';
import { Play, Loader2, RefreshCw, Database, Network, RotateCcw, Globe, Github, Blend, BookOpen, GitCompare, Calculator, Ruler } from 'lucide-react';
import { cn } from './lib/utils';
import * as tf from '@tensorflow/tfjs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PCA } from 'ml-pca';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import { translations } from './translations';

const Plot = createPlotlyComponent(Plotly);

const mnistData = new MnistData();
const modelService = new AutoencoderService();
const EMBEDDING_DIM = 8;

function drawImageToCanvas(imageData: Float32Array, canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const imgData = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const val = imageData[i] * 255;
    imgData.data[i * 4] = val;     // R
    imgData.data[i * 4 + 1] = val; // G
    imgData.data[i * 4 + 2] = val; // B
    imgData.data[i * 4 + 3] = 255; // Alpha
  }
  
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  offscreen.getContext('2d')?.putImageData(imgData, 0, 0);
  
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreen, 0, 0, 28, 28, 0, 0, canvas.width, canvas.height);
}

// --- Componentes Visuales ---

function TabularDataViewer({ images, selectedIndex, onSelect, lang }: { images: Float32Array[], selectedIndex: number, onSelect: (idx: number) => void, lang: 'en' | 'es' }) {
  const [colOffset, setColOffset] = useState(0);
  const visibleCols = 15;
  const maxOffset = 784 - visibleCols;

  if (!images.length) return null;
  
  const colIndices = Array.from({length: visibleCols}, (_, i) => i + colOffset);
  const t = translations[lang];

  return (
    <div className="w-full flex flex-col gap-3">
      <div className="flex items-center gap-4 bg-zinc-900/50 p-3 rounded-xl border border-zinc-800">
        <span className="text-xs font-medium text-zinc-400 whitespace-nowrap">{t.s1_shift}</span>
        <input 
          type="range" 
          min="0" 
          max={maxOffset} 
          value={colOffset} 
          onChange={(e) => setColOffset(parseInt(e.target.value))}
          className="flex-1 h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
        />
        <span className="text-xs font-mono text-emerald-400 w-32 text-right">
          {t.s1_pixels} {colOffset} - {colOffset + visibleCols - 1}
        </span>
      </div>
      <div className="w-full max-h-[400px] overflow-y-auto text-xs font-mono bg-zinc-950 rounded-xl border border-zinc-800 shadow-inner relative [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-zinc-950 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-thumb]:rounded-full">
        <table className="w-full text-left border-collapse whitespace-nowrap">
          <thead className="sticky top-0 bg-zinc-950 z-10 shadow-md">
            <tr className="text-zinc-500 border-b border-zinc-800">
              <th className="p-3 font-medium bg-zinc-950 sticky left-0 z-20 border-r border-zinc-800">{t.s1_sample}</th>
              {colIndices.map(colIdx => (
                <th key={colIdx} className={cn("p-3 font-medium", colIdx === 392 ? "text-emerald-400" : "")}>
                  Píxel_{colIdx}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {images.map((img, i) => (
              <tr 
                key={i} 
                onClick={() => onSelect(i)}
                className={cn(
                  "border-b border-zinc-800/50 transition-colors cursor-pointer",
                  selectedIndex === i ? "bg-emerald-900/20 text-emerald-100" : "text-zinc-300 hover:bg-zinc-900/50"
                )}
              >
                <td className={cn("p-3 font-semibold bg-zinc-950/90 sticky left-0 border-r border-zinc-800", selectedIndex === i ? "text-emerald-400" : "text-zinc-400")}>Img {i + 1}</td>
                {colIndices.map(colIdx => (
                  <td key={colIdx} className={cn("p-3", colIdx === 392 ? "text-emerald-400 bg-emerald-500/10 rounded" : "")}>
                    {img[colIdx].toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ArchitectureDiagram({ lang }: { lang: 'en' | 'es' }) {
  const t = translations[lang];
  const layers = [
    { name: 'Input', size: 784, act: '-', h: 'h-40', color: 'bg-blue-500/10 border-blue-500/50 text-blue-400' },
    { name: 'Dense', size: 256, act: 'ReLU', h: 'h-32', color: 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400' },
    { name: 'Dense', size: 128, act: 'ReLU', h: 'h-24', color: 'bg-violet-500/10 border-violet-500/50 text-violet-400' },
    { name: 'Dense', size: 64, act: 'ReLU', h: 'h-16', color: 'bg-fuchsia-500/10 border-fuchsia-500/50 text-fuchsia-400' },
    { name: 'Embedding', size: 8, act: 'Linear', h: 'h-8', color: 'bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.3)]' },
    { name: 'Dense', size: 64, act: 'ReLU', h: 'h-16', color: 'bg-fuchsia-500/10 border-fuchsia-500/50 text-fuchsia-400' },
    { name: 'Dense', size: 128, act: 'ReLU', h: 'h-24', color: 'bg-violet-500/10 border-violet-500/50 text-violet-400' },
    { name: 'Dense', size: 256, act: 'ReLU', h: 'h-32', color: 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400' },
    { name: 'Output', size: 784, act: 'Sigmoid', h: 'h-40', color: 'bg-blue-500/10 border-blue-500/50 text-blue-400' },
  ];

  return (
    <div className="flex items-center justify-center gap-1 md:gap-3 p-6 bg-zinc-900/50 rounded-xl border border-zinc-800 h-full overflow-x-auto">
      {layers.map((l, i) => (
        <div key={i} className="flex flex-col items-center gap-2 min-w-[48px]">
          <div className="text-[9px] text-zinc-500 uppercase tracking-wider font-semibold text-center h-6">{l.name}</div>
          <div className={cn(
            "w-8 md:w-12 border-2 rounded-md transition-all flex items-center justify-center relative group", 
            l.h, l.color
          )}>
            <span className="text-[10px] font-mono rotate-90 md:rotate-0">{l.size}</span>
            <div className="absolute -top-10 bg-zinc-800 text-zinc-200 text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
              {l.size} {t.s2_neurons}
            </div>
          </div>
          <div className="text-[9px] text-zinc-400 font-mono mt-1 bg-zinc-950 px-1.5 py-0.5 rounded border border-zinc-800">{l.act}</div>
        </div>
      ))}
    </div>
  );
}

// --- Aplicación Principal ---

export default function App() {
  const [lang, setLang] = useState<'en' | 'es'>('en');
  const t = translations[lang];

  const [isLoaded, setIsLoaded] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  
  const [epoch, setEpoch] = useState(0);
  const [targetEpochs, setTargetEpochs] = useState(50);
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number, val_loss: number}[]>([]);
  
  const [testImages, setTestImages] = useState<Float32Array[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [embedding, setEmbedding] = useState<number[]>(new Array(EMBEDDING_DIM).fill(0));
  const [reconstructed, setReconstructed] = useState<Float32Array | null>(null);
  
  const [mapData, setMapData] = useState<{x: number[], y: number[], z: number[], images: Float32Array[]} | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Interpolation State
  const [interpA, setInterpA] = useState(0);
  const [interpB, setInterpB] = useState(1);
  const [interpWeight, setInterpWeight] = useState(0.5);
  const [interpRecon, setInterpRecon] = useState<Float32Array | null>(null);

  // Comparator State
  const [compA, setCompA] = useState(0);
  const [compB, setCompB] = useState(1);
  const [embCompA, setEmbCompA] = useState<number[]>(new Array(EMBEDDING_DIM).fill(0));
  const [embCompB, setEmbCompB] = useState<number[]>(new Array(EMBEDDING_DIM).fill(0));
  const [compDistanceMetric, setCompDistanceMetric] = useState<'euclidean' | 'cosine'>('euclidean');
  const [compDistance, setCompDistance] = useState<number>(0);

  // Operator State
  const [opA, setOpA] = useState(0);
  const [opB, setOpB] = useState(1);
  const [opC, setOpC] = useState(2);
  const [opResult, setOpResult] = useState<Float32Array | null>(null);
  const [opResultEmb, setOpResultEmb] = useState<number[]>(new Array(EMBEDDING_DIM).fill(0));
  const [opDistanceMetric, setOpDistanceMetric] = useState<'euclidean' | 'cosine'>('euclidean');
  const [opDistance, setOpDistance] = useState<number>(0);

  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const reconstructedCanvasRef = useRef<HTMLCanvasElement>(null);
  const interpCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // Refs to access latest state inside training callback
  const selectedIndexRef = useRef(0);
  const testImagesRef = useRef<Float32Array[]>([]);

  const calculateInitialLoss = async () => {
    const trainData = mnistData.getTrainData(4000);
    const testData = mnistData.getTestData(1000);
    const trainLoss = await modelService.evaluate(trainData);
    const valLoss = await modelService.evaluate(testData);
    if (trainLoss !== null && valLoss !== null) {
      setLossHistory([{ epoch: 0, loss: trainLoss, val_loss: valLoss }]);
    }
    trainData.xs.dispose();
    trainData.ys.dispose();
    testData.xs.dispose();
    testData.ys.dispose();
  };

  useEffect(() => {
    // Load data on mount
    mnistData.load().then(() => {
      setIsLoaded(true);
      const testData = mnistData.getTestData(50);
      const images: Float32Array[] = [];
      const dataSync = testData.xs.dataSync();
      for (let i = 0; i < 50; i++) {
        images.push(new Float32Array(dataSync.slice(i * 784, (i + 1) * 784)));
      }
      setTestImages(images);
      testImagesRef.current = images;
      testData.xs.dispose();
      testData.ys.dispose();

      // Initialize model with random weights so users can see the "before training" state
      modelService.buildModel(EMBEDDING_DIM);
      setIsInitialized(true);
      calculateInitialLoss();
    });
  }, []);

  // When initialized, select the first image to show random noise reconstruction
  useEffect(() => {
    if (isInitialized && testImages.length > 0) {
      handleSelectImage(0);
    }
  }, [isInitialized]);

  // Interpolation Effect
  useEffect(() => {
    if (!isInitialized || !testImages[interpA] || !testImages[interpB] || !modelService.encoder) return;
    
    // Get embeddings for A and B
    const embA = modelService.getEmbedding(tf.tensor2d(testImages[interpA], [1, 784]));
    const embB = modelService.getEmbedding(tf.tensor2d(testImages[interpB], [1, 784]));
    
    const arrA = Array.from(embA);
    const arrB = Array.from(embB);
    
    // Linear interpolation
    const mixed = arrA.map((a, i) => a * (1 - interpWeight) + arrB[i] * interpWeight);
    
    // Reconstruct
    const recon = modelService.reconstructFromEmbedding(new Float32Array(mixed));
    setInterpRecon(recon);
    
    if (interpCanvasRef.current) {
      drawImageToCanvas(recon, interpCanvasRef.current);
    }
  }, [interpA, interpB, interpWeight, isInitialized, isTrained, testImages]);

  // Helper for distance calculation
  const calculateDistance = (vecA: number[], vecB: number[], metric: 'euclidean' | 'cosine') => {
    if (metric === 'euclidean') {
      return Math.sqrt(vecA.reduce((sum, a, i) => sum + Math.pow(a - vecB[i], 2), 0));
    } else {
      // Cosine Similarity
      let dotProduct = 0;
      let normA = 0;
      let normB = 0;
      for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
      }
      if (normA === 0 || normB === 0) return 0;
      return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
  };

  // Comparator Effect
  useEffect(() => {
    if (!isInitialized || !modelService.encoder || !testImages[compA] || !testImages[compB]) return;
    const embA = Array.from(modelService.getEmbedding(tf.tensor2d(testImages[compA], [1, 784])));
    const embB = Array.from(modelService.getEmbedding(tf.tensor2d(testImages[compB], [1, 784])));
    setEmbCompA(embA);
    setEmbCompB(embB);
    setCompDistance(calculateDistance(embA, embB, compDistanceMetric));
  }, [compA, compB, compDistanceMetric, isInitialized, isTrained, testImages]);

  // Operator Effect
  useEffect(() => {
    if (!isInitialized || !modelService.encoder || !testImages[opA] || !testImages[opB] || !testImages[opC]) return;
    const embA = Array.from(modelService.getEmbedding(tf.tensor2d(testImages[opA], [1, 784])));
    const embB = Array.from(modelService.getEmbedding(tf.tensor2d(testImages[opB], [1, 784])));
    const embC = Array.from(modelService.getEmbedding(tf.tensor2d(testImages[opC], [1, 784])));

    const resultEmb = embA.map((a, i) => a - embB[i] + embC[i]);
    setOpResultEmb(resultEmb);
    const recon = modelService.reconstructFromEmbedding(new Float32Array(resultEmb));
    setOpResult(recon);
    
    // Calculate distance between the original Image A and the Result
    setOpDistance(calculateDistance(embA, resultEmb, opDistanceMetric));
  }, [opA, opB, opC, opDistanceMetric, isInitialized, isTrained, testImages]);

  const resetModel = async () => {
    if (isTraining) return;
    modelService.buildModel(EMBEDDING_DIM);
    setEpoch(0);
    setLossHistory([]);
    setIsTrained(false);
    handleSelectImage(selectedIndexRef.current);
    await calculateInitialLoss();
    setMapData(null);
  };

  const trainModel = async () => {
    setIsTraining(true);
    setIsTrained(false);
    
    const trainData = mnistData.getTrainData(4000);
    const testData = mnistData.getTestData(1000);
    
    await modelService.train(trainData, testData, targetEpochs, async (ep, logs) => {
      setEpoch(prev => prev + 1);
      setLossHistory(prev => [...prev, { epoch: prev.length + 1, loss: logs.loss, val_loss: logs.val_loss }]);
      
      // Update the reconstruction live!
      updateLiveReconstruction(selectedIndexRef.current);
    });
    
    trainData.xs.dispose();
    trainData.ys.dispose();
    testData.xs.dispose();
    testData.ys.dispose();
    
    setIsTraining(false);
    setIsTrained(true);
  };

  const handleSelectImage = (index: number) => {
    setSelectedIndex(index);
    selectedIndexRef.current = index;
    updateLiveReconstruction(index);
  };

  const updateLiveReconstruction = (index: number) => {
    if (!modelService.encoder || !modelService.decoder) return;
    const imgData = testImagesRef.current[index];
    if (!imgData) return;
    
    if (originalCanvasRef.current) {
      drawImageToCanvas(imgData, originalCanvasRef.current);
    }
    
    // Get embedding
    const tensor = tf.tensor2d(imgData, [1, 784]);
    const emb = modelService.getEmbedding(tensor);
    const embArray = Array.from(emb);
    setEmbedding(embArray);
    
    // Reconstruct
    const recon = modelService.reconstructFromEmbedding(new Float32Array(embArray));
    setReconstructed(recon);
    if (reconstructedCanvasRef.current) {
      drawImageToCanvas(recon, reconstructedCanvasRef.current);
    }
  };

  const handleSliderChange = (dimIndex: number, value: number) => {
    const newEmb = [...embedding];
    newEmb[dimIndex] = value;
    setEmbedding(newEmb);
    
    const recon = modelService.reconstructFromEmbedding(new Float32Array(newEmb));
    setReconstructed(recon);
    if (reconstructedCanvasRef.current) {
      drawImageToCanvas(recon, reconstructedCanvasRef.current);
    }
  };

  const generate3DMap = async () => {
    if (!modelService.encoder) return;
    
    // Get 500 images for the map
    const testData = mnistData.getTestData(500);
    const dataSync = testData.xs.dataSync();
    const images: Float32Array[] = [];
    for (let i = 0; i < 500; i++) {
      images.push(new Float32Array(dataSync.slice(i * 784, (i + 1) * 784)));
    }
    
    // Get embeddings
    const embeddingsTensor = modelService.encoder.predict(testData.xs) as tf.Tensor;
    const embeddingsArray = await embeddingsTensor.array() as number[][];
    
    // Run PCA
    const pca = new PCA(embeddingsArray);
    const reduced = pca.predict(embeddingsArray, { nComponents: 3 }).to2DArray();
    
    setMapData({
      x: reduced.map(r => r[0]),
      y: reduced.map(r => r[1]),
      z: reduced.map(r => r[2]),
      images
    });
    
    testData.xs.dispose();
    testData.ys.dispose();
    embeddingsTensor.dispose();
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans flex flex-col">
      <header className="border-b border-zinc-800 p-6 bg-zinc-900/30">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-white">{t.title}</h1>
            <p className="text-zinc-400 mt-2 max-w-3xl">
              {t.subtitle}
            </p>
          </div>
          <div className="flex items-center gap-2 bg-zinc-950 p-1 rounded-lg border border-zinc-800">
            <button 
              onClick={() => setLang('en')}
              className={cn("px-3 py-1 text-sm font-medium rounded-md transition-colors", lang === 'en' ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-zinc-200")}
            >
              EN
            </button>
            <button 
              onClick={() => setLang('es')}
              className={cn("px-3 py-1 text-sm font-medium rounded-md transition-colors", lang === 'es' ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-zinc-200")}
            >
              ES
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-6 flex flex-col gap-12">
        
        {/* Intro Section */}
        <section className="space-y-6 bg-zinc-900/20 p-8 rounded-2xl border border-zinc-800/50">
          <div className="flex items-center gap-2 text-xl font-medium text-emerald-400">
            <BookOpen className="w-6 h-6" />
            <h2>{t.intro_title}</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-3">
              <h3 className="text-lg font-medium text-zinc-200">{t.intro_emb_title}</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">{t.intro_emb_text}</p>
            </div>
            <div className="space-y-3">
              <h3 className="text-lg font-medium text-zinc-200">{t.intro_ae_title}</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">{t.intro_ae_text}</p>
            </div>
          </div>
        </section>

        {/* Section 1: Data (Wide) */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
            <Database className="w-5 h-5 text-emerald-500" />
            <h2>{t.s1_title}</h2>
          </div>
          <p className="text-sm text-zinc-400">
            {t.s1_desc}
          </p>
          <div className="bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 text-sm text-zinc-400">
            {t.s1_dataset_info}
          </div>
          <div className="bg-emerald-900/10 border border-emerald-900/30 p-3 rounded-lg text-xs text-emerald-200/80">
            <strong>{t.s1_norm}</strong> {t.s1_norm_desc}
          </div>
          
          {isLoaded ? (
            <div className="flex flex-col lg:flex-row gap-6 items-start">
              <div className="flex-1 w-full overflow-hidden">
                <TabularDataViewer 
                  images={testImages} 
                  selectedIndex={selectedIndex} 
                  onSelect={handleSelectImage} 
                  lang={lang}
                />
              </div>
              <div className="w-full lg:w-64 flex flex-col items-center gap-4 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800">
                <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">{t.s1_preview}</h3>
                <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800 shadow-inner">
                  {testImages[selectedIndex] && (
                    <MiniCanvas imageData={testImages[selectedIndex]} className="w-32 h-32" />
                  )}
                </div>
                <p className="text-xs text-zinc-500 text-center">
                  {t.s1_row_selected} {selectedIndex + 1}
                </p>
              </div>
            </div>
          ) : (
            <div className="h-48 bg-zinc-900/50 rounded-xl border border-zinc-800 flex items-center justify-center">
              <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
            </div>
          )}
        </section>

        {/* Section 2: Architecture (Wide) */}
        <section className="space-y-4 border-t border-zinc-800 pt-12">
          <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
            <Network className="w-5 h-5 text-emerald-500" />
            <h2>{t.s2_title}</h2>
          </div>
          <p className="text-sm text-zinc-400">
            {t.s2_desc}
          </p>
          <ArchitectureDiagram lang={lang} />
        </section>

        {/* Section 3: Playground (Merged Training & Latent Space) */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-12">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
              <Play className="w-5 h-5 text-emerald-500" />
              <h2>{t.s3_title}</h2>
            </div>
            
            {/* Controls (TF Playground style) */}
            <div className="flex flex-wrap items-center gap-4 bg-zinc-900/50 p-2 rounded-2xl border border-zinc-800">
              <button
                onClick={resetModel}
                disabled={!isLoaded || isTraining}
                className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-xl transition-colors disabled:opacity-50"
                title="Reset network weights"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
              
              <button
                onClick={trainModel}
                disabled={!isLoaded || isTraining}
                className={cn(
                  "flex items-center justify-center gap-2 py-2 px-6 rounded-xl font-medium transition-colors shadow-lg",
                  !isLoaded || isTraining 
                    ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                    : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20"
                )}
              >
                {isTraining ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> {t.s3_training_btn}</>
                ) : (
                  <><Play className="w-4 h-4" /> {t.s3_train_btn}</>
                )}
              </button>

              <div className="flex items-center gap-3 px-4 border-l border-zinc-800">
                <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{t.s3_epochs}</span>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="10"
                  value={targetEpochs}
                  onChange={(e) => setTargetEpochs(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="w-24 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:opacity-50"
                />
                <span className="text-sm font-mono text-emerald-400 w-8">{targetEpochs}</span>
              </div>
              
              <div className="flex items-center gap-2 px-4 border-l border-zinc-800 min-w-[120px]">
                <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{t.s3_current_epoch}</span>
                <span className="text-lg font-mono text-white">{epoch.toString().padStart(3, '0')}</span>
              </div>
              
              {lossHistory.length > 0 && (
                <div className="flex items-center gap-4 px-4 border-l border-zinc-800">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-medium text-emerald-500 uppercase tracking-wider">{t.s3_loss_train}</span>
                    <span className="text-sm font-mono text-white">{lossHistory[lossHistory.length - 1].loss.toFixed(4)}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] font-medium text-indigo-400 uppercase tracking-wider">{t.s3_loss_val}</span>
                    <span className="text-sm font-mono text-white">{lossHistory[lossHistory.length - 1].val_loss.toFixed(4)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Main Playground Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            
            {/* Left Column: Image Selection & Input */}
            <div className="xl:col-span-3 space-y-6 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 flex flex-col items-center">
              <div className="w-full space-y-1 text-center mb-2">
                <h3 className="font-medium text-zinc-200">{t.s3_input}</h3>
                <p className="text-xs text-zinc-500">{t.s3_input_desc}</p>
              </div>
              
              <div className="p-3 bg-zinc-950 rounded-2xl border border-zinc-800 shadow-xl">
                <canvas ref={originalCanvasRef} width={140} height={140} className="bg-black rounded-lg" />
              </div>
              
              <div className="w-full grid grid-cols-5 gap-1.5 max-h-[220px] overflow-y-auto p-1 pr-2 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-zinc-900/50 [&::-webkit-scrollbar-thumb]:bg-zinc-700 [&::-webkit-scrollbar-thumb]:rounded-full">
                {testImages.map((img, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSelectImage(idx)}
                    className={cn(
                      "aspect-square rounded-md overflow-hidden border-2 transition-all",
                      selectedIndex === idx ? "border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]" : "border-transparent hover:border-zinc-700",
                    )}
                  >
                    <MiniCanvas imageData={img} />
                  </button>
                ))}
              </div>
            </div>

            {/* Center Column: Latent Space & Chart */}
            <div className="xl:col-span-6 space-y-6 flex flex-col">
              {/* Latent Space Sliders */}
              <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 flex-1 flex flex-col">
                <div className="text-center space-y-1 mb-6">
                  <h3 className="font-medium text-emerald-400">{t.s3_latent}</h3>
                  <p className="text-xs text-zinc-500">
                    {isTrained || epoch > 0 
                      ? t.s3_latent_desc_trained
                      : t.s3_latent_desc_untrained}
                  </p>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4">
                  {embedding.map((val, idx) => (
                    <div key={idx} className="flex items-center gap-3 group">
                      <span className="text-[10px] font-mono text-zinc-500 w-4 text-right group-hover:text-emerald-400 transition-colors">D{idx}</span>
                      <input
                        type="range"
                        min="-8"
                        max="8"
                        step="0.01"
                        value={val}
                        onChange={(e) => handleSliderChange(idx, parseFloat(e.target.value))}
                        className="flex-1 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                      />
                      <span className="text-[10px] font-mono text-zinc-400 w-10 text-right bg-zinc-950 px-1.5 py-1 rounded">
                        {val.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Chart */}
              <div className="bg-zinc-900/30 p-4 rounded-2xl border border-zinc-800 h-48 flex flex-col">
                <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2 ml-2">{t.s3_chart_title}</h3>
                <div className="flex-1 w-full bg-zinc-950 rounded-xl border border-zinc-800 p-2">
                  {lossHistory.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={lossHistory} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#71717a" fontSize={11} tickLine={false} axisLine={false} />
                        <YAxis stroke="#71717a" fontSize={11} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
                          itemStyle={{ fontSize: '12px' }}
                          labelStyle={{ color: '#a1a1aa', fontSize: '12px', marginBottom: '4px' }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="#34d399" strokeWidth={2} dot={lossHistory.length === 1} name="Train" />
                        <Line type="monotone" dataKey="val_loss" stroke="#818cf8" strokeWidth={2} dot={lossHistory.length === 1} name="Val" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-zinc-600 text-xs">
                      {t.s3_chart_empty}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: Reconstructed Output */}
            <div className="xl:col-span-3 space-y-6 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 flex flex-col items-center justify-center">
              <div className="w-full space-y-1 text-center mb-2">
                <h3 className="font-medium text-zinc-200">{t.s3_output}</h3>
                <p className="text-xs text-zinc-500">{t.s3_output_desc}</p>
              </div>
              
              <div className="p-3 bg-zinc-950 rounded-2xl border border-zinc-800 shadow-xl relative group">
                <canvas ref={reconstructedCanvasRef} width={180} height={180} className="bg-black rounded-lg" />
                <div className="absolute inset-0 border-2 border-emerald-500/0 group-hover:border-emerald-500/50 rounded-2xl transition-colors pointer-events-none" />
              </div>
              
              <div className="text-center px-4">
                <p className="text-xs text-zinc-500">
                  {t.s3_output_note}
                </p>
              </div>
            </div>

          </div>
        </section>

        {/* Section 4: 3D Latent Space Map */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-12">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
              <Network className="w-5 h-5 text-emerald-500" />
              <h2>{t.s4_title}</h2>
            </div>
            <button
              onClick={generate3DMap}
              disabled={!isLoaded || isTraining}
              className={cn(
                "flex items-center justify-center gap-2 py-2 px-6 rounded-xl font-medium transition-colors shadow-lg",
                !isLoaded || isTraining 
                  ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                  : "bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/20"
              )}
            >
              <RefreshCw className="w-4 h-4" /> {t.s4_btn}
            </button>
          </div>
          
          <p className="text-sm text-zinc-400 max-w-3xl">
            {t.s4_desc}
          </p>

          <div className="flex flex-col lg:flex-row gap-6">
            <div className="flex-1 h-[600px] bg-zinc-900/30 rounded-2xl border border-zinc-800 relative overflow-hidden flex items-center justify-center">
              {!mapData ? (
                <div className="text-zinc-500 flex flex-col items-center gap-2">
                  <Network className="w-8 h-8 opacity-50" />
                  <p>{t.s4_empty}</p>
                </div>
              ) : (
                <Plot
                  data={[
                    {
                      x: mapData.x,
                      y: mapData.y,
                      z: mapData.z,
                      type: 'scatter3d',
                      mode: 'markers',
                      marker: {
                        size: 4,
                        color: mapData.z,
                        colorscale: 'Viridis',
                        opacity: 0.8
                      },
                      hoverinfo: 'none'
                    }
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 0, r: 0, b: 0, t: 0 },
                    paper_bgcolor: 'transparent',
                    scene: {
                      xaxis: { title: 'PCA 1', backgroundcolor: 'transparent', gridcolor: '#27272a', zerolinecolor: '#3f3f46' },
                      yaxis: { title: 'PCA 2', backgroundcolor: 'transparent', gridcolor: '#27272a', zerolinecolor: '#3f3f46' },
                      zaxis: { title: 'PCA 3', backgroundcolor: 'transparent', gridcolor: '#27272a', zerolinecolor: '#3f3f46' },
                    }
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%', height: '100%' }}
                  onHover={(e) => {
                    const pt = e.points[0];
                    if (pt && pt.pointNumber !== undefined) {
                      setHoveredIndex(pt.pointNumber);
                    }
                  }}
                  onUnhover={() => setHoveredIndex(null)}
                />
              )}
            </div>
            
            {/* Hover Side Panel */}
            <div className="w-full lg:w-64 flex flex-col items-center gap-4 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider text-center">{t.s4_hover}</h3>
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800 shadow-inner w-full aspect-square flex items-center justify-center">
                {hoveredIndex !== null && mapData ? (
                  <MiniCanvas key={`hover-${hoveredIndex}`} imageData={mapData.images[hoveredIndex]} className="w-full h-full rounded-lg" />
                ) : (
                  <span className="text-zinc-600 text-xs text-center px-4">{t.s4_hover_desc}</span>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Section: Embedding Comparator */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
              <GitCompare className="w-5 h-5 text-emerald-500" />
              <h2>{t.s_comp_title}</h2>
            </div>
            <div className="flex items-center gap-2 bg-zinc-900/50 p-2 rounded-lg border border-zinc-800">
              <Ruler className="w-4 h-4 text-zinc-400" />
              <select 
                value={compDistanceMetric} 
                onChange={e => setCompDistanceMetric(e.target.value as 'euclidean' | 'cosine')}
                className="bg-transparent text-xs text-zinc-300 outline-none cursor-pointer"
              >
                <option value="euclidean">{lang === 'en' ? 'Euclidean Distance' : 'Distancia Euclidiana'}</option>
                <option value="cosine">{lang === 'en' ? 'Cosine Similarity' : 'Similitud Coseno'}</option>
              </select>
              <div className="px-2 py-1 bg-zinc-950 rounded text-xs font-mono text-emerald-400 border border-zinc-800">
                {compDistance.toFixed(4)}
              </div>
            </div>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl">{t.s_comp_desc}</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Comp A */}
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 flex flex-col items-center gap-6">
              <div className="flex flex-col items-center gap-2">
                <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                  {testImages[compA] && <MiniCanvas imageData={testImages[compA]} className="w-32 h-32 rounded-lg" />}
                </div>
                <select value={compA} onChange={e => setCompA(Number(e.target.value))} className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none">
                  {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
                </select>
              </div>
              <EmbeddingBarChart embedding={embCompA} />
            </div>

            {/* Comp B */}
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 flex flex-col items-center gap-6">
              <div className="flex flex-col items-center gap-2">
                <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                  {testImages[compB] && <MiniCanvas imageData={testImages[compB]} className="w-32 h-32 rounded-lg" />}
                </div>
                <select value={compB} onChange={e => setCompB(Number(e.target.value))} className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none">
                  {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
                </select>
              </div>
              <EmbeddingBarChart embedding={embCompB} />
            </div>
          </div>
        </section>

        {/* Section: Latent Arithmetic */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
              <Calculator className="w-5 h-5 text-emerald-500" />
              <h2>{t.s_op_title}</h2>
            </div>
            <div className="flex items-center gap-2 bg-zinc-900/50 p-2 rounded-lg border border-zinc-800">
              <span className="text-xs text-zinc-500">{lang === 'en' ? 'Distance (A vs Result):' : 'Distancia (A vs Resultado):'}</span>
              <Ruler className="w-4 h-4 text-zinc-400" />
              <select 
                value={opDistanceMetric} 
                onChange={e => setOpDistanceMetric(e.target.value as 'euclidean' | 'cosine')}
                className="bg-transparent text-xs text-zinc-300 outline-none cursor-pointer"
              >
                <option value="euclidean">{lang === 'en' ? 'Euclidean' : 'Euclidiana'}</option>
                <option value="cosine">{lang === 'en' ? 'Cosine' : 'Coseno'}</option>
              </select>
              <div className="px-2 py-1 bg-zinc-950 rounded text-xs font-mono text-emerald-400 border border-zinc-800">
                {opDistance.toFixed(4)}
              </div>
            </div>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl">{t.s_op_desc}</p>

          <div className="flex flex-col md:flex-row items-center justify-center gap-4 bg-zinc-900/30 p-8 rounded-2xl border border-zinc-800 overflow-x-auto">
            {/* A */}
            <div className="flex flex-col items-center gap-2">
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                {testImages[opA] && <MiniCanvas imageData={testImages[opA]} className="w-20 h-20 rounded-lg" />}
              </div>
              <select value={opA} onChange={e => setOpA(Number(e.target.value))} className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none">
                {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
              </select>
            </div>

            <div className="text-2xl font-mono text-zinc-500">-</div>

            {/* B */}
            <div className="flex flex-col items-center gap-2">
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                {testImages[opB] && <MiniCanvas imageData={testImages[opB]} className="w-20 h-20 rounded-lg" />}
              </div>
              <select value={opB} onChange={e => setOpB(Number(e.target.value))} className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none">
                {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
              </select>
            </div>

            <div className="text-2xl font-mono text-zinc-500">+</div>

            {/* C */}
            <div className="flex flex-col items-center gap-2">
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                {testImages[opC] && <MiniCanvas imageData={testImages[opC]} className="w-20 h-20 rounded-lg" />}
              </div>
              <select value={opC} onChange={e => setOpC(Number(e.target.value))} className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none">
                {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
              </select>
            </div>

            <div className="text-2xl font-mono text-zinc-500">=</div>

            {/* Result */}
            <div className="flex flex-col items-center gap-2">
              <div className="p-2 bg-zinc-950 rounded-xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
                {opResult ? (
                  <MiniCanvas key={`op-${opA}-${opB}-${opC}`} imageData={opResult} className="w-20 h-20 rounded-lg" />
                ) : (
                  <div className="w-20 h-20 bg-black rounded-lg" />
                )}
              </div>
              <span className="text-xs font-medium text-emerald-400 py-1">{t.s_op_result}</span>
            </div>
          </div>
        </section>

        {/* Section 5: Interpolation (Morphing) */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-24">
          <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
            <Blend className="w-5 h-5 text-emerald-500" />
            <h2>{t.s5_title}</h2>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl">{t.s5_desc}</p>
          
          <div className="flex flex-col md:flex-row items-center justify-center gap-8 bg-zinc-900/30 p-8 rounded-2xl border border-zinc-800">
            {/* Image A */}
            <div className="flex flex-col items-center gap-4">
              <h3 className="text-sm font-medium text-zinc-400">{t.s5_img_a}</h3>
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                {testImages[interpA] && <MiniCanvas imageData={testImages[interpA]} className="w-24 h-24 rounded-lg" />}
              </div>
              <select 
                value={interpA} 
                onChange={e => setInterpA(Number(e.target.value))}
                className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none"
              >
                {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
              </select>
            </div>

            {/* Slider & Morph */}
            <div className="flex-1 w-full max-w-md flex flex-col items-center gap-6">
              <div className="w-full flex items-center gap-4">
                <span className="text-xs font-mono text-zinc-500">A</span>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.01" 
                  value={interpWeight} 
                  onChange={e => setInterpWeight(Number(e.target.value))} 
                  className="flex-1 h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                />
                <span className="text-xs font-mono text-zinc-500">B</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <h3 className="text-sm font-medium text-emerald-400">{t.s5_morph}</h3>
                <div className="p-3 bg-zinc-950 rounded-2xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
                  <canvas ref={interpCanvasRef} width={120} height={120} className="bg-black rounded-lg" />
                </div>
              </div>
            </div>

            {/* Image B */}
            <div className="flex flex-col items-center gap-4">
              <h3 className="text-sm font-medium text-zinc-400">{t.s5_img_b}</h3>
              <div className="p-2 bg-zinc-950 rounded-xl border border-zinc-800">
                {testImages[interpB] && <MiniCanvas imageData={testImages[interpB]} className="w-24 h-24 rounded-lg" />}
              </div>
              <select 
                value={interpB} 
                onChange={e => setInterpB(Number(e.target.value))}
                className="bg-zinc-800 text-xs text-zinc-300 rounded px-2 py-1 outline-none"
              >
                {testImages.map((_, i) => <option key={i} value={i}>Img {i+1}</option>)}
              </select>
            </div>
          </div>
          
          <div className="bg-zinc-900/40 p-5 rounded-xl border border-zinc-800 text-sm text-zinc-400 mt-4 leading-relaxed">
            <strong className="text-zinc-200 block mb-2">{t.s5_interp_info.split('?')[0]}?</strong>
            {t.s5_interp_info.split('?')[1]}
          </div>
        </section>

        {/* Section 6: Student Activities */}
        <section className="space-y-6 border-t border-zinc-800 pt-12 pb-24">
          <div className="flex items-center gap-2 text-lg font-medium text-zinc-200">
            <BookOpen className="w-5 h-5 text-emerald-500" />
            <h2>{t.s6_title}</h2>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl">{t.s6_desc}</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q1_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q1_text}</p>
            </div>
            
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q2_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q2_text}</p>
            </div>
            
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q3_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q3_text}</p>
            </div>
            
            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q4_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q4_text}</p>
            </div>
            
            <div className="md:col-span-2 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q5_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q5_text}</p>
            </div>
            
            <div className="md:col-span-2 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q6_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q6_text}</p>
            </div>

            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q7_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q7_text}</p>
            </div>

            <div className="bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q8_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q8_text}</p>
            </div>

            <div className="md:col-span-2 bg-zinc-900/30 p-6 rounded-2xl border border-zinc-800 hover:border-emerald-500/30 transition-colors">
              <h3 className="text-sm font-medium text-emerald-400 mb-2">{t.s6_q9_title}</h3>
              <p className="text-sm text-zinc-300">{t.s6_q9_text}</p>
            </div>
          </div>
        </section>

        {/* References Section */}
        <section className="space-y-4 border-t border-zinc-800 pt-12 pb-8">
          <h2 className="text-lg font-medium text-zinc-200">{t.ref_title}</h2>
          <ul className="list-disc list-inside space-y-2 text-sm text-zinc-400">
            <li>{t.ref_1}</li>
            <li>
              <a href="https://github.com/zalandoresearch/fashion-mnist" target="_blank" rel="noreferrer" className="text-emerald-500 hover:underline">
                {t.ref_2}
              </a>
            </li>
          </ul>
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 py-8 text-center flex flex-col items-center justify-center gap-3 bg-zinc-950">
        <p className="text-zinc-400 text-sm">
          {t.footer_created} <span className="text-zinc-200 font-medium">David Díaz {t.footer_role}</span>
        </p>
        <p className="text-zinc-500 text-xs italic">
          {t.footer_ai}
        </p>
        <div className="flex items-center gap-6 text-sm">
          <a href="https://daviddiazsolis.com" target="_blank" rel="noreferrer" className="text-emerald-500 hover:text-emerald-400 transition-colors flex items-center gap-1.5">
            <Globe className="w-4 h-4" /> daviddiazsolis.com
          </a>
          <a href="https://github.com/daviddiazsolis" target="_blank" rel="noreferrer" className="text-emerald-500 hover:text-emerald-400 transition-colors flex items-center gap-1.5">
            <Github className="w-4 h-4" /> GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

// Helper component for the small gallery
function MiniCanvas({ imageData, className }: { imageData: Float32Array, className?: string, key?: React.Key }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current && imageData) {
      drawImageToCanvas(imageData, canvasRef.current);
    }
  }, [imageData]);

  return (
    <canvas 
      ref={canvasRef} 
      width={112} 
      height={112} 
      className={cn("w-full h-full bg-black object-cover", className)}
    />
  );
}

// Helper component for embedding bars
function EmbeddingBarChart({ embedding }: { embedding: number[] }) {
  const maxVal = Math.max(8, ...embedding.map(Math.abs)); // scale to at least 8

  return (
    <div className="flex flex-col gap-1 w-full max-w-xs">
      {embedding.map((val, idx) => {
        const width = Math.min(100, (Math.abs(val) / maxVal) * 100);
        const isPositive = val >= 0;
        return (
          <div key={idx} className="flex items-center gap-2 text-xs font-mono">
            <span className="text-zinc-500 w-4">D{idx}</span>
            <div className="flex-1 flex items-center">
              {/* Negative side */}
              <div className="flex-1 flex justify-end">
                {!isPositive && (
                  <div className="h-2 bg-rose-500 rounded-l-sm" style={{ width: `${width}%` }} />
                )}
              </div>
              {/* Center line */}
              <div className="w-px h-4 bg-zinc-700 mx-1" />
              {/* Positive side */}
              <div className="flex-1 flex justify-start">
                {isPositive && (
                  <div className="h-2 bg-emerald-500 rounded-r-sm" style={{ width: `${width}%` }} />
                )}
              </div>
            </div>
            <span className="text-zinc-400 w-8 text-right">{val.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

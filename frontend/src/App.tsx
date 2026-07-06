import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import { Compass, RotateCcw, AlertTriangle, Eye, ShieldAlert, Cpu } from 'lucide-react'

// Components
import CinematicBackground from './components/CinematicBackground'
import UploadCard from './components/UploadCard'
import LoadingTimeline from './components/LoadingTimeline'
import ExplainabilityViewer from './components/ExplainabilityViewer'
import BirdProfileCard from './components/BirdProfileCard'
import FieldGuide from './components/FieldGuide'
import ChatSection from './components/ChatSection'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

interface PredictionItem {
  species_id: string
  common_name: string
  confidence: number
}

interface PredictResponse {
  species_id: string
  common_name: string
  confidence: number
  top_predictions: PredictionItem[]
  gradcam_image: string
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [speciesData, setSpeciesData] = useState<any | null>(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Ref attached to the top of the results container so we can scroll
  // exactly there — and nowhere else — after a successful prediction.
  const resultsRef = useRef<HTMLDivElement>(null)

  // Single, intentional scroll: fires once when both prediction and
  // speciesData are ready. requestAnimationFrame defers the scroll
  // until after the results DOM has been painted, so the measurement
  // is accurate and the animation is already in progress.
  useEffect(() => {
    if (prediction && speciesData) {
      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      })
    }
  }, [prediction, speciesData])

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setIsAnalyzing(true)
    setError(null)
    setPrediction(null)
    setSpeciesData(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // 1. Classify image & get Grad-CAM
      const predictRes = await axios.post<PredictResponse>(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setPrediction(predictRes.data)

      // 2. Retrieve structured field guide data
      const speciesRes = await axios.get(`${API_URL}/species/${predictRes.data.species_id}`)
      setSpeciesData(speciesRes.data)
      
    } catch (err: any) {
      console.error(err)
      setError(
        err.response?.data?.detail || 
        "A connection error occurred. Please verify that the deep learning server is active."
      )
      setSelectedFile(null)
      setPreviewUrl(null)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setPrediction(null)
    setSpeciesData(null)
    setError(null)
    setShowGradcam(true)
  }

  return (
    <div className="relative min-h-screen flex flex-col justify-between overflow-x-hidden font-sans">
      
      {/* Background Graphic elements */}
      <CinematicBackground />

      {/* Main Container */}
      <main className="relative flex-grow container mx-auto px-4 py-8 md:py-16 z-10 flex flex-col justify-center">
        <AnimatePresence mode="wait">
          
          {/* SCREEN 1: Hero & Upload */}
          {!selectedFile && !isAnalyzing && !prediction && (
            <motion.div
              key="hero-screen"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center space-y-12 max-w-3xl mx-auto py-12"
            >
              {/* Header Editorial */}
              <div className="space-y-4">
                <span className="font-mono text-xs uppercase tracking-[0.25em] text-gold-400">
                  Artificial Intelligence Field Journal
                </span>
                <h1 className="font-serif text-5xl md:text-7xl font-bold tracking-tight text-ivory-100 leading-none">
                  Bird Identifier AI
                </h1>
                <p className="font-serif italic text-lg md:text-xl text-ivory-300/80 max-w-xl mx-auto font-light leading-relaxed">
                  Every feather tells a story. Let's discover this one's.
                </p>
              </div>

              {/* Polaroid natural drop zone */}
              <UploadCard 
                onImageSelected={handleImageSelected} 
                isAnalyzing={isAnalyzing} 
              />
            </motion.div>
          )}

          {/* SCREEN 2: Analyzing Timeline Loader */}
          {isAnalyzing && (
            <motion.div
              key="loader-screen"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="py-12"
            >
              <LoadingTimeline stage="predict" />
            </motion.div>
          )}

          {/* SCREEN 3: Prediction Results & Field Guide */}
          {prediction && speciesData && previewUrl && (
            <motion.div
              ref={resultsRef}
              key="results-screen"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              className="space-y-16 py-6"
            >
              {/* Header bar with Reset action */}
              <div className="flex items-center justify-between border-b border-forest-800/40 pb-6">
                <span className="font-mono text-xs uppercase tracking-widest text-gold-400 flex items-center gap-2">
                  <Compass className="w-4 h-4 animate-spin-slow" />
                  Specimen Identified Successfully
                </span>
                <button
                  onClick={handleReset}
                  className="flex items-center space-x-2 text-xs uppercase tracking-widest text-ivory-300 hover:text-gold-400 transition-colors duration-300 font-mono"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>Log Another</span>
                </button>
              </div>

              {/* Explainability split slider */}
              {showGradcam && (
                <div className="flex justify-center">
                  <ExplainabilityViewer 
                    originalImage={previewUrl} 
                    gradcamImage={prediction.gradcam_image} 
                  />
                </div>
              )}

              {/* Species card profiles summaries */}
              <BirdProfileCard
                commonName={prediction.common_name}
                scientificName={speciesData.scientific_name}
                confidence={prediction.confidence}
                conservationStatus={speciesData.conservation.status}
                showGradcam={showGradcam}
                onToggleGradcam={() => setShowGradcam(!showGradcam)}
              />

              {/* Split layout: Top alternative predictions & specs */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left columns: alternative classifier predictions */}
                <div className="lg:col-span-1 bg-forest-900/10 border border-forest-800/40 p-6 rounded-3xl space-y-6">
                  <h4 className="font-mono text-xs uppercase tracking-widest text-gold-400 border-b border-forest-800/40 pb-2">
                    Classification Details
                  </h4>
                  <div className="space-y-4">
                    <p className="text-xs text-ivory-300/60 leading-relaxed font-light">
                      The Convolutional Neural Network evaluated alternative feature hierarchies. Below are the top-5 candidate rankings and confidence values.
                    </p>
                    <div className="space-y-3">
                      {prediction.top_predictions.map((alt, idx) => (
                        <div 
                          key={alt.species_id} 
                          className={`flex items-center justify-between p-3 rounded-xl border ${
                            idx === 0 
                              ? 'bg-gold-500/10 border-gold-500/30' 
                              : 'bg-forest-950/40 border-forest-900'
                          }`}
                        >
                          <div className="text-sm">
                            <span className="font-mono text-xs text-gold-400 mr-2">#{idx + 1}</span>
                            <span className={idx === 0 ? 'font-medium text-ivory-100' : 'text-ivory-300 font-light'}>
                              {alt.common_name.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <span className="font-mono text-xs text-ivory-200">
                            {Math.round(alt.confidence * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right columns: Brief introduction specs snippet */}
                <div className="lg:col-span-2 bg-forest-900/10 border border-forest-800/40 p-6 rounded-3xl flex flex-col justify-between">
                  <div className="space-y-4">
                    <h4 className="font-mono text-xs uppercase tracking-widest text-gold-400 border-b border-forest-800/40 pb-2">
                      Specimen Context
                    </h4>
                    <p className="font-serif italic text-lg leading-relaxed text-ivory-200/90 font-light">
                      "Found primarily in {speciesData.habitat.toLowerCase()}, this species of the {speciesData.family} family showcases distinct behaviors related to {speciesData.behaviour.split('.')[0].toLowerCase()}."
                    </p>
                  </div>
                  <div className="text-xs text-ivory-300/40 border-t border-forest-800/40 pt-4 mt-6">
                    Vector Database Retrieval: Grounded on {speciesData.interesting_facts.length} verified records in the local field guide.
                  </div>
                </div>
              </div>

              {/* Scroll guide separator */}
              <div className="flex flex-col items-center justify-center space-y-2 opacity-40 select-none">
                <span className="text-xs uppercase tracking-widest font-mono text-ivory-300">Scroll to view Journal</span>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5 animate-bounce">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                </svg>
              </div>

              {/* Field Guide Page Spread */}
              <FieldGuide data={speciesData} />

              {/* AI Naturalist Assistant Chat Section */}
              <ChatSection
                speciesId={prediction.species_id}
                commonName={prediction.common_name}
                apiUrl={API_URL}
              />
            </motion.div>
          )}

          {/* Error Boundary */}
          {error && (
            <motion.div
              key="error-screen"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-md mx-auto text-center space-y-6 bg-red-950/20 border border-red-500/20 p-8 rounded-3xl backdrop-blur-md"
            >
              <div className="w-16 h-16 bg-red-900/30 rounded-2xl flex items-center justify-center text-red-400 mx-auto border border-red-500/30">
                <AlertTriangle className="w-8 h-8 stroke-[1.2]" />
              </div>
              <div className="space-y-2">
                <h3 className="font-serif text-2xl text-ivory-100">Inference Interrupted</h3>
                <p className="text-sm text-red-200/60 leading-relaxed font-light">{error}</p>
              </div>
              <button
                onClick={handleReset}
                className="bg-forest-800 hover:bg-forest-700 text-ivory-100 text-xs font-mono uppercase tracking-widest px-6 py-3 rounded-xl border border-forest-700 transition-all shadow-md active:scale-95"
              >
                Return to Desk
              </button>
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* Footer copyright editorial */}
      <footer className="relative border-t border-forest-900/40 py-6 text-center text-xs text-ivory-400/20 font-mono tracking-widest uppercase z-10">
        © 2026 Bird Identifier AI · BBC Earth & National Geographic Styled
      </footer>
    </div>
  )
}

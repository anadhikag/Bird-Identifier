import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, HelpCircle, Eye, ShieldAlert, Cpu } from 'lucide-react'

interface LoadingTimelineProps {
  stage: 'predict' | 'chat'
}

const PREDICT_STEPS = [
  "Analyzing feather patterns...",
  "Running deep neural network classification...",
  "Generating Grad-CAM attention heatmap...",
  "Retrieving field guide records from vector DB...",
  "Formulating naturalist study report..."
]

const CHAT_STEPS = [
  "Reading ornithologist notes...",
  "Consulting the knowledge base...",
  "Grounding context for accuracy...",
  "Drafting answer details..."
]

export default function LoadingTimeline({ stage }: LoadingTimelineProps) {
  const steps = stage === 'predict' ? PREDICT_STEPS : CHAT_STEPS
  const [currentStepIdx, setCurrentStepIdx] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStepIdx((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1
        }
        return prev
      })
    }, 2200)

    return () => clearInterval(interval)
  }, [steps])

  return (
    <div className="flex flex-col items-center justify-center p-8 text-center space-y-8 max-w-md mx-auto">
      {/* Immersive nature-inspired pulsing loader */}
      <div className="relative flex items-center justify-center w-28 h-28">
        <motion.div
          animate={{
            scale: [1, 1.15, 1],
            rotate: 360,
          }}
          transition={{
            duration: 12,
            ease: "easeInOut",
            repeat: Infinity,
          }}
          className="absolute inset-0 rounded-full border border-forest-500/20 border-t-gold-500/80 border-b-moss-500/60"
        />
        
        <motion.div
          animate={{
            scale: [0.9, 1.05, 0.9],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 4,
            ease: "easeInOut",
            repeat: Infinity,
          }}
          className="absolute w-20 h-20 rounded-full bg-forest-900/40 blur-xl"
        />

        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          className="text-gold-400 z-10"
        >
          {stage === 'predict' ? (
            <Cpu className="w-8 h-8 stroke-[1.2] opacity-80" />
          ) : (
            <BookOpen className="w-8 h-8 stroke-[1.2] opacity-80" />
          )}
        </motion.div>
      </div>

      {/* Typing steps display */}
      <div className="space-y-4 min-h-[90px] flex flex-col justify-center">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStepIdx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.5 }}
            className="text-lg font-serif text-ivory-200"
          >
            {steps[currentStepIdx]}
          </motion.div>
        </AnimatePresence>

        {/* Bullet timelines */}
        <div className="flex justify-center space-x-2">
          {steps.map((_, idx) => (
            <div
              key={idx}
              className={`h-1.5 rounded-full transition-all duration-700 ${
                idx === currentStepIdx
                  ? 'w-6 bg-gold-500'
                  : idx < currentStepIdx
                  ? 'w-2 bg-moss-600'
                  : 'w-2 bg-forest-900/50 border border-forest-500/20'
              }`}
            />
          ))}
        </div>
      </div>
      
      <p className="text-xs text-ivory-400/40 font-mono tracking-wider uppercase">
        {stage === 'predict' ? 'Stage I Classification' : 'Consulting Assistant'}
      </p>
    </div>
  )
}

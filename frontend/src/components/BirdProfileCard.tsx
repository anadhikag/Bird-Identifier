import React from 'react'
import { Award, Compass, Shield, ToggleLeft, ToggleRight } from 'lucide-react'
import { motion } from 'framer-motion'

interface BirdProfileCardProps {
  commonName: string
  scientificName: string
  confidence: number
  conservationStatus: string
  showGradcam: boolean
  onToggleGradcam: () => void
}

export default function BirdProfileCard({
  commonName,
  scientificName,
  confidence,
  conservationStatus,
  showGradcam,
  onToggleGradcam,
}: BirdProfileCardProps) {
  
  // Format conservation badge color
  const getConservationStyle = (status: string) => {
    const s = status.toLowerCase()
    if (s.includes('least concern') || s.includes('lc')) {
      return { bg: 'bg-emerald-950/60 border-emerald-500/30 text-emerald-400', label: 'Least Concern' }
    }
    if (s.includes('threatened') || s.includes('nt') || s.includes('near')) {
      return { bg: 'bg-yellow-950/60 border-yellow-500/30 text-yellow-400', label: 'Near Threatened' }
    }
    if (s.includes('vulnerable') || s.includes('vu')) {
      return { bg: 'bg-orange-950/60 border-orange-500/30 text-orange-400', label: 'Vulnerable' }
    }
    if (s.includes('endangered') || s.includes('en')) {
      return { bg: 'bg-red-950/60 border-red-500/30 text-red-400', label: 'Endangered' }
    }
    return { bg: 'bg-gold-950/60 border-gold-500/30 text-gold-400', label: status || 'Unknown Status' }
  }

  const statusStyle = getConservationStyle(conservationStatus)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.2 }}
      className="w-full bg-forest-900/20 backdrop-blur-md rounded-3xl p-6 md:p-8 border border-forest-800/80 shadow-xl"
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        
        {/* Left: Species Header Details */}
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-xs uppercase tracking-widest text-gold-400/80 flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5" />
              Primary Identification
            </span>
            <span className={`text-xs font-mono uppercase px-2.5 py-0.5 rounded-full border ${statusStyle.bg}`}>
              {statusStyle.label}
            </span>
          </div>

          <div>
            <h2 className="font-serif text-3xl md:text-4xl text-ivory-100 font-bold tracking-tight">
              {commonName}
            </h2>
            <p className="font-serif italic text-lg text-gold-400/90 mt-1">
              {scientificName}
            </p>
          </div>
        </div>

        {/* Right: Stats & Interaction */}
        <div className="flex flex-wrap items-center gap-6 md:self-end">
          {/* Confidence Ring */}
          <div className="flex items-center space-x-3">
            <div className="relative flex items-center justify-center w-16 h-16 rounded-full border border-forest-700/60 bg-forest-950/40">
              {/* Circular progress overlay */}
              <svg className="absolute w-full h-full -rotate-90">
                <circle
                  cx="32"
                  cy="32"
                  r="26"
                  className="stroke-forest-800/40"
                  strokeWidth="3"
                  fill="transparent"
                />
                <circle
                  cx="32"
                  cy="32"
                  r="26"
                  className="stroke-gold-500"
                  strokeWidth="3"
                  fill="transparent"
                  strokeDasharray={163.36}
                  strokeDashoffset={163.36 - (163.36 * confidence)}
                />
              </svg>
              <span className="font-mono text-sm text-ivory-200">
                {Math.round(confidence * 100)}%
              </span>
            </div>
            <div>
              <span className="block text-xs uppercase tracking-widest text-ivory-400/50 font-mono">
                Confidence
              </span>
              <span className="text-sm font-medium text-ivory-200 flex items-center gap-1">
                <Award className="w-4 h-4 text-gold-400" />
                Verified Match
              </span>
            </div>
          </div>

          {/* Vertical divider */}
          <div className="hidden sm:block w-[1px] h-12 bg-forest-800/60"></div>

          {/* Gradcam toggle button */}
          <button
            onClick={onToggleGradcam}
            className="flex items-center space-x-3 bg-forest-950/40 hover:bg-forest-900/60 border border-forest-800 hover:border-gold-500/50 px-4 py-2.5 rounded-2xl transition-all duration-300 shadow-md group"
          >
            <span className="text-sm text-ivory-300 font-light group-hover:text-ivory-100 transition-colors">
              Grad-CAM Visualizer
            </span>
            {showGradcam ? (
              <ToggleRight className="w-7 h-7 text-gold-500" />
            ) : (
              <ToggleLeft className="w-7 h-7 text-ivory-400" />
            )}
          </button>
        </div>

      </div>
    </motion.div>
  )
}

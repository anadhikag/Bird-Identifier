import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Eye, HelpCircle } from 'lucide-react'

interface ExplainabilityViewerProps {
  originalImage: string // Base64 or object URL of the uploaded image
  gradcamImage: string  // Base64 string of the Grad-CAM overlay
}

export default function ExplainabilityViewer({ originalImage, gradcamImage }: ExplainabilityViewerProps) {
  const [sliderPosition, setSliderPosition] = useState(50) // Percentage (0-100)
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPosition(percentage)
  }

  const handleTouchMove = (e: TouchEvent) => {
    if (!isResizing) return
    handleMove(e.touches[0].clientX)
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizing) return
    handleMove(e.clientX)
  }

  const handleMouseUp = () => {
    setIsResizing(false)
  }

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      window.addEventListener('touchmove', handleTouchMove)
      window.addEventListener('touchend', handleMouseUp)
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('touchend', handleMouseUp)
    }
  }, [isResizing])

  const handleMouseDown = () => {
    setIsResizing(true)
  }

  return (
    <div className="space-y-4">
      {/* Description */}
      <div className="flex items-start space-x-3 bg-forest-900/40 p-4 rounded-2xl border border-forest-800/60 max-w-xl">
        <Eye className="w-5 h-5 text-gold-400 mt-0.5 flex-shrink-0" />
        <div className="text-sm text-ivory-300 leading-relaxed font-light">
          <strong className="text-gold-400 font-medium">Explainability View:</strong> Slide between the original bird photograph and the Grad-CAM heatmap overlay. The crimson regions highlight the specific features (crest, eye shape, beak thickness) the neural network prioritized to classify the species.
        </div>
      </div>

      {/* Slider Container */}
      <div 
        ref={containerRef}
        className="relative w-full max-w-xl mx-auto aspect-square md:aspect-[4/3] rounded-3xl overflow-hidden shadow-2xl border border-forest-800 select-none group"
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
      >
        {/* Underlay: Original image */}
        <img 
          src={originalImage} 
          alt="Original uploaded bird" 
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />
        <div className="absolute top-4 left-4 bg-forest-950/80 backdrop-blur-md px-3 py-1 rounded-full text-xs font-mono tracking-wider text-ivory-200 uppercase z-20 shadow-md">
          Original Photo
        </div>

        {/* Overlay: Grad-CAM heatmap */}
        <div 
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          <img 
            src={gradcamImage.startsWith('data:') ? gradcamImage : `data:image/png;base64,${gradcamImage}`} 
            alt="Grad-CAM activation overlay" 
            className="absolute inset-0 w-full h-full object-cover max-w-none"
            style={{ width: containerRef.current?.getBoundingClientRect().width }}
          />
          <div className="absolute top-4 right-4 bg-gold-600/90 backdrop-blur-md px-3 py-1 rounded-full text-xs font-mono tracking-wider text-ivory-100 uppercase z-20 shadow-md">
            Grad-CAM Heatmap
          </div>
        </div>

        {/* Vertical divider line */}
        <div 
          className="absolute top-0 bottom-0 w-1 bg-gold-400/80 cursor-ew-resize z-30"
          style={{ left: `${sliderPosition}%` }}
        >
          {/* Slider handle handle */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-gold-500 border-2 border-ivory-100 text-ivory-100 flex items-center justify-center shadow-2xl transition-transform group-hover:scale-110 active:scale-95">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
              <path d="m18 8 4 4-4 4M6 8l-4 4 4 4" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  )
}

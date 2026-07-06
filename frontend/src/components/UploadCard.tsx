import React, { useState, useRef } from 'react'
import { Camera, Image as ImageIcon, Upload } from 'lucide-react'
import { motion } from 'framer-motion'

interface UploadCardProps {
  onImageSelected: (file: File) => void
  isAnalyzing: boolean
}

export default function UploadCard({ onImageSelected, isAnalyzing }: UploadCardProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true)
    } else if (e.type === "dragleave") {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      if (file.type.startsWith('image/')) {
        onImageSelected(file)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelected(e.target.files[0])
    }
  }

  const onButtonClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
      className="w-full max-w-lg mx-auto"
    >
      <div 
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`relative cursor-pointer transition-all duration-500 rounded-3xl p-6 md:p-8
          ${isDragActive 
            ? 'bg-moss-100/10 border-moss-500 scale-[1.02]' 
            : 'bg-ivory-100/5 hover:bg-ivory-100/10 border-ivory-300/20'
          } border-2 border-dashed shadow-2xl backdrop-blur-md`}
        onClick={onButtonClick}
      >
        <input 
          ref={fileInputRef}
          type="file" 
          className="hidden" 
          accept="image/*"
          onChange={handleFileChange}
          disabled={isAnalyzing}
        />

        <div className="flex flex-col items-center justify-center text-center py-10 md:py-14 space-y-6">
          {/* Naturalist journal styling inside */}
          <div className="relative w-24 h-24 rounded-2xl bg-forest-900/50 flex items-center justify-center border border-forest-500/30 text-gold-400 group-hover:text-gold-300 transition-colors shadow-inner">
            <motion.div
              animate={isDragActive ? { y: [0, -10, 0] } : {}}
              transition={{ repeat: Infinity, duration: 1.5 }}
            >
              <Camera className="w-10 h-10 stroke-[1.2]" />
            </motion.div>
            <div className="absolute -bottom-2 -right-2 bg-moss-600 rounded-lg p-1.5 border border-forest-600 shadow-md">
              <Upload className="w-4 h-4 text-ivory-100" />
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="font-serif text-2xl text-ivory-100">
              Place a photograph here
            </h3>
            <p className="text-sm text-ivory-300/70 font-light max-w-xs leading-relaxed">
              Drag and drop your bird photo or click to explore your local archives.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs text-gold-400/80 font-mono tracking-widest uppercase">
            <span>JPEG</span>
            <span className="w-1.5 h-1.5 rounded-full bg-forest-500"></span>
            <span>PNG</span>
            <span className="w-1.5 h-1.5 rounded-full bg-forest-500"></span>
            <span>WEBP</span>
          </div>
        </div>

        {/* Cinematic photo-desk corners */}
        <div className="absolute top-4 left-4 w-4 h-4 border-t-2 border-l-2 border-ivory-300/30 rounded-tl-lg"></div>
        <div className="absolute top-4 right-4 w-4 h-4 border-t-2 border-r-2 border-ivory-300/30 rounded-tr-lg"></div>
        <div className="absolute bottom-4 left-4 w-4 h-4 border-b-2 border-l-2 border-ivory-300/30 rounded-bl-lg"></div>
        <div className="absolute bottom-4 right-4 w-4 h-4 border-b-2 border-r-2 border-ivory-300/30 rounded-br-lg"></div>
      </div>
    </motion.div>
  )
}

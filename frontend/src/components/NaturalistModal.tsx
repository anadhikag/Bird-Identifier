import React, { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Github, Linkedin, Feather } from 'lucide-react'

interface NaturalistModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function NaturalistModal({ isOpen, onClose }: NaturalistModalProps) {
  const closeRef = useRef<HTMLButtonElement>(null)

  // Close on ESC
  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  // Prevent body scroll while open
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  // Focus close button when opened for accessibility
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => closeRef.current?.focus(), 80)
    }
  }, [isOpen])

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="fixed inset-0 z-50 bg-forest-950/80 backdrop-blur-sm"
            onClick={onClose}
            aria-hidden="true"
          />

          {/* Panel */}
          <motion.div
            key="panel"
            role="dialog"
            aria-modal="true"
            aria-label="Meet the Naturalist"
            initial={{ opacity: 0, y: 32, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.97 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="fixed z-50 inset-0 flex items-center justify-center pointer-events-none px-4"
          >
            <div
              className="pointer-events-auto relative w-full max-w-lg bg-forest-950 border border-forest-800/60 rounded-3xl shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Subtle top gradient accent */}
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-gold-500/50 to-transparent" />

              {/* Close button */}
              <button
                ref={closeRef}
                onClick={onClose}
                className="absolute top-5 right-5 text-ivory-400/40 hover:text-ivory-100 transition-colors duration-200 z-10"
                aria-label="Close"
              >
                <X className="w-5 h-5" strokeWidth={1.5} />
              </button>

              {/* Content */}
              <div className="px-8 pt-10 pb-8 flex flex-col gap-7">

                {/* Eye-mark — small feather icon section label */}
                <div className="flex items-center gap-3">
                  <Feather className="w-3.5 h-3.5 text-gold-400" strokeWidth={1.5} />
                  <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-gold-400">
                    The Naturalist
                  </span>
                </div>

                {/* Name + subtitle */}
                <div className="space-y-2">
                  <h2 className="font-serif text-3xl font-bold text-ivory-100 leading-tight">
                    Anadhika Goswami
                  </h2>
                  <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ivory-300/50">
                    AI Engineering · Computer Vision · Machine Learning
                  </p>
                </div>

                {/* Hairline botanical divider */}
                <div className="flex items-center gap-4">
                  <div className="flex-1 h-px bg-forest-800/60" />
                  {/* Tiny feather ornament */}
                  <svg viewBox="0 0 24 24" className="w-4 h-4 text-gold-500/40" fill="currentColor">
                    <path d="M20.84 2.18a.5.5 0 0 0-.64-.05C17.18 4.3 14 8.18 12.44 12.4l-1.06-1.06a.5.5 0 0 0-.7.7l1.35 1.35c-.3.9-.5 1.8-.6 2.7l-1.42-1.42a.5.5 0 0 0-.7.7l1.75 1.76c-.06.8-.04 1.6.07 2.38a.5.5 0 0 0 .49.42h.08a.5.5 0 0 0 .41-.57c-.1-.68-.12-1.38-.06-2.09l1.3 1.3a.5.5 0 0 0 .7-.7l-1.6-1.6c.1-.8.28-1.6.55-2.38l1.72 1.72a.5.5 0 0 0 .7-.7L13.6 13.3C15.1 9.3 18.1 5.6 20.9 3.1a.5.5 0 0 0-.06-.92z"/>
                  </svg>
                  <div className="flex-1 h-px bg-forest-800/60" />
                </div>

                {/* Description */}
                <p className="font-serif italic text-base text-ivory-200/80 leading-relaxed font-light">
                  Hi! I'm Anadhika Goswami, a Computer Science student specialising in Data Science and AI. I built{' '}
                  <span className="text-gold-400 not-italic font-normal">Birdie</span> to combine deep learning, Grad-CAM
                  explainability, retrieval-augmented generation (RAG), and my knowledge of cloud (Microsoft Azure) into
                  an interactive experience inspired by natural history field guides.
                </p>

                {/* CTA buttons */}
                <div className="flex flex-col sm:flex-row gap-3 pt-1">
                  <a
                    href="https://github.com/anadhikag"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center justify-center gap-2.5 flex-1 px-5 py-3 rounded-xl border border-forest-700 bg-forest-900/60 hover:border-gold-500/50 hover:bg-forest-800/80 transition-all duration-300 text-ivory-200 hover:text-ivory-100 text-xs font-mono uppercase tracking-widest"
                  >
                    <Github className="w-4 h-4 group-hover:scale-110 transition-transform duration-200" strokeWidth={1.5} />
                    GitHub
                  </a>
                  <a
                    href="https://in.linkedin.com/in/anadhika-goswami"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center justify-center gap-2.5 flex-1 px-5 py-3 rounded-xl bg-gold-500/15 hover:bg-gold-500/25 border border-gold-500/30 hover:border-gold-400/60 transition-all duration-300 text-gold-300 hover:text-gold-200 text-xs font-mono uppercase tracking-widest"
                  >
                    <Linkedin className="w-4 h-4 group-hover:scale-110 transition-transform duration-200" strokeWidth={1.5} />
                    LinkedIn
                  </a>
                </div>

                {/* Bottom note */}
                <p className="text-center text-[11px] text-ivory-400/25 font-mono tracking-wider pt-1">
                  Built with ❤️ for birds, AI and curious minds.
                </p>
              </div>

              {/* Subtle bottom gradient accent */}
              <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-forest-700/30 to-transparent" />
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

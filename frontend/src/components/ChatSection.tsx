import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Feather, NotebookText, HelpCircle } from 'lucide-react'
import axios from 'axios'
import LoadingTimeline from './LoadingTimeline'

interface ChatMessage {
  id: string
  question: string
  answer: string
  isTypingDone: boolean
}

interface ChatSectionProps {
  speciesId: string
  commonName: string
  apiUrl: string
}

export default function ChatSection({ speciesId, commonName, apiUrl }: ChatSectionProps) {
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const listEndRef = useRef<HTMLDivElement>(null)

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || isLoading) return

    const userQuestion = question.trim()
    setQuestion('')
    setIsLoading(true)

    try {
      const response = await axios.post(`${apiUrl}/chat`, {
        species_id: speciesId,   // canonical folder ID, e.g. "017.Cardinal"
        question: userQuestion,
      })

      const newMsg: ChatMessage = {
        id: Math.random().toString(36).substring(7),
        question: userQuestion,
        answer: response.data.answer,
        isTypingDone: false,
      }

      setChatHistory((prev) => [...prev, newMsg])
    } catch (err) {
      console.error(err)
      const errorMsg: ChatMessage = {
        id: Math.random().toString(36).substring(7),
        question: userQuestion,
        answer: "I apologize, but my connection to the field journals failed. Please ensure the backend is available.",
        isTypingDone: true,
      }
      setChatHistory((prev) => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  // Scroll to the bottom of the chat history ONLY after the user has
  // submitted at least one question. Guarding with chatHistory.length > 0
  // prevents the effect from firing on initial mount (React always runs
  // every useEffect once when the component first appears), which was the
  // cause of the page auto-scrolling to the very bottom on prediction.
  useEffect(() => {
    if (chatHistory.length > 0) {
      listEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatHistory])

  return (
    <div className="w-full max-w-4xl mx-auto space-y-10 border-t border-forest-800/40 pt-16 pb-20">
      
      {/* Editorial Title */}
      <div className="text-center space-y-3">
        <span className="font-mono text-xs uppercase tracking-widest text-gold-400/80 flex items-center justify-center gap-1.5">
          <Feather className="w-3.5 h-3.5" />
          Interactive Field Assistant
        </span>
        <h3 className="font-serif text-3xl md:text-4xl text-ivory-100 font-bold">
          Ask this bird anything
        </h3>
        <p className="text-sm text-ivory-300/60 font-light max-w-md mx-auto leading-relaxed">
          Query details regarding its feeding habits, unique vocalizations, or nesting behaviors to consult the retrieved archive records.
        </p>
      </div>

      {/* History Area */}
      <div className="space-y-12">
        <AnimatePresence>
          {chatHistory.map((item) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="space-y-4 max-w-3xl mx-auto"
            >
              {/* Question: Clean editorial label */}
              <div className="flex items-start space-x-3">
                <div className="font-mono text-xs text-gold-400 mt-1 uppercase tracking-wider select-none">Q:</div>
                <h4 className="font-serif text-lg font-semibold text-ivory-200">
                  {item.question}
                </h4>
              </div>

              {/* Answer: Ornithologist notebook text */}
              <div className="flex items-start space-x-3 bg-forest-900/10 p-5 rounded-2xl border border-forest-800/30">
                <div className="font-mono text-xs text-moss-500 mt-1 uppercase tracking-wider select-none">Notes:</div>
                <div className="font-serif text-base text-ivory-300 leading-relaxed font-light italic max-w-none">
                  <TypewriterText 
                    text={item.answer} 
                    onComplete={() => {
                      item.isTypingDone = true
                    }} 
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading status */}
        {isLoading && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="py-6"
          >
            <LoadingTimeline stage="chat" />
          </motion.div>
        )}
        
        <div ref={listEndRef} />
      </div>

      {/* Minimal Editorial Input Box */}
      <form 
        onSubmit={handleAsk}
        className="relative max-w-2xl mx-auto border-b border-ivory-300/30 focus-within:border-gold-500 transition-colors duration-500 pb-3 flex items-center"
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={`Inquire about the ${commonNameCleanup(commonName)}...`}
          disabled={isLoading}
          className="w-full bg-transparent text-lg text-ivory-100 font-serif italic placeholder:text-ivory-400/30 focus:outline-none pr-12 pl-2"
        />
        
        <button
          type="submit"
          disabled={!question.trim() || isLoading}
          className="absolute right-2 p-2 text-ivory-400 hover:text-gold-400 disabled:text-ivory-400/20 disabled:hover:text-ivory-400/20 transition-all duration-300"
        >
          <Send className="w-5 h-5 stroke-[1.5]" />
        </button>
      </form>

    </div>
  )
}

// Typewriter Component
function TypewriterText({ text, onComplete }: { text: string; onComplete: () => void }) {
  const [displayText, setDisplayText] = useState('')
  
  useEffect(() => {
    let index = 0
    let currentStr = ''
    setDisplayText('')
    
    // Slow typing speed
    const interval = setInterval(() => {
      if (index < text.length) {
        currentStr += text[index]
        setDisplayText(currentStr)
        index++
      } else {
        clearInterval(interval)
        onComplete()
      }
    }, 15) // Speed controls character placement

    return () => clearInterval(interval)
  }, [text])

  return <span>{displayText}</span>
}

function commonNameCleanup(name: string): string {
  if (!name) return ''
  if (name.includes('.')) {
    name = name.split('.', 2)[1]
  }
  return name.replace(/_/g, ' ')
}

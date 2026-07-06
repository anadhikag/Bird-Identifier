import React from 'react'
import { motion } from 'framer-motion'
import { Map, Wind, Trees, Apple, Info, Shield, HelpCircle } from 'lucide-react'

interface FieldGuideProps {
  data: {
    common_name: string
    scientific_name: string
    family: string
    order: string
    identification: {
      length: string
      wingspan: string
      weight: string
      differences: string
      features: string
    }
    habitat: string
    geographic_distribution: string
    migration: {
      status: string
      pattern: string
      raw: string
    }
    diet: string
    behaviour: string
    vocalization: string
    breeding: string
    conservation: {
      status: string
      threats: string
      raw: string
    }
    ecological_importance: string
    interesting_facts: string[]
  }
}

export default function FieldGuide({ data }: FieldGuideProps) {
  const {
    common_name,
    scientific_name,
    family,
    order,
    identification,
    habitat,
    geographic_distribution,
    migration,
    diet,
    behaviour,
    vocalization,
    breeding,
    conservation,
    ecological_importance,
    interesting_facts,
  } = data

  return (
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 1 }}
      className="field-paper rounded-[2.5rem] overflow-hidden p-6 md:p-12 lg:p-16 max-w-5xl mx-auto space-y-12 md:space-y-16"
    >
      {/* Editorial Title / Header */}
      <div className="border-b border-bark-200 pb-8 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <span className="font-mono text-xs uppercase tracking-widest text-moss-600 block mb-2">
            Naturalist Field Guide Entry
          </span>
          <h2 className="font-serif text-4xl md:text-5xl font-bold tracking-tight text-bark-900">
            {commonNameCleanup(common_name)}
          </h2>
          <p className="font-serif italic text-xl text-bark-600/80 mt-1">
            {scientific_name}
          </p>
        </div>
        <div className="flex flex-col md:items-end text-xs font-mono uppercase text-bark-500 space-y-1.5">
          <div><strong className="text-bark-700">Family:</strong> {family || 'N/A'}</div>
          <div><strong className="text-bark-700">Order:</strong> {order || 'N/A'}</div>
        </div>
      </div>

      {/* Main Grid Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12">
        {/* Left column: Quick Stats & Identification details */}
        <div className="md:col-span-1 bg-ivory-200/50 p-6 rounded-2xl border border-ivory-300/40 h-fit space-y-6">
          <h4 className="font-serif text-lg font-bold uppercase tracking-wider text-bark-800 border-b border-bark-200 pb-2">
            Field Specs
          </h4>
          
          <div className="space-y-4 font-sans text-sm text-bark-700">
            <div>
              <span className="block text-xs uppercase font-mono tracking-widest text-bark-500 mb-0.5">Average Length</span>
              <p className="font-medium text-bark-800">{identification.length || 'N/A'}</p>
            </div>
            <div>
              <span className="block text-xs uppercase font-mono tracking-widest text-bark-500 mb-0.5">Average Wingspan</span>
              <p className="font-medium text-bark-800">{identification.wingspan || 'N/A'}</p>
            </div>
            <div>
              <span className="block text-xs uppercase font-mono tracking-widest text-bark-500 mb-0.5">Average Weight</span>
              <p className="font-medium text-bark-800">{identification.weight || 'N/A'}</p>
            </div>
            {identification.features && (
              <div>
                <span className="block text-xs uppercase font-mono tracking-widest text-bark-500 mb-0.5">Key Features</span>
                <p className="leading-relaxed font-light">{identification.features}</p>
              </div>
            )}
            {identification.differences && (
              <div>
                <span className="block text-xs uppercase font-mono tracking-widest text-bark-500 mb-0.5">Sex Dimorphism</span>
                <p className="leading-relaxed font-light text-xs bg-ivory-300/40 p-2.5 rounded-lg italic">
                  {identification.differences}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right Columns: Description Spread */}
        <div className="md:col-span-2 space-y-8 md:space-y-10">
          
          {/* Habitat & Distribution */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h3 className="font-serif text-lg font-bold text-bark-900 flex items-center gap-2">
                <Trees className="w-5 h-5 text-moss-600 stroke-[1.5]" />
                Habitat
              </h3>
              <p className="text-sm font-light leading-relaxed text-bark-700">
                {habitat || 'Information pending details.'}
              </p>
            </div>

            <div className="space-y-2">
              <h3 className="font-serif text-lg font-bold text-bark-900 flex items-center gap-2">
                <Map className="w-5 h-5 text-moss-600 stroke-[1.5]" />
                Geographic Distribution
              </h3>
              <p className="text-sm font-light leading-relaxed text-bark-700">
                {geographic_distribution || 'Information pending details.'}
              </p>
            </div>
          </div>

          {/* Migration & Diet */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 border-t border-bark-200/50 pt-8">
            <div className="space-y-2">
              <h3 className="font-serif text-lg font-bold text-bark-900 flex items-center gap-2">
                <Wind className="w-5 h-5 text-moss-600 stroke-[1.5]" />
                Migration
              </h3>
              <div className="text-sm font-light leading-relaxed text-bark-700 space-y-1">
                {migration.status && (
                  <p><strong className="font-medium text-bark-800">Status:</strong> {migration.status}</p>
                )}
                <p>{migration.pattern || migration.raw || 'No notable migration patterns recorded.'}</p>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="font-serif text-lg font-bold text-bark-900 flex items-center gap-2">
                <Apple className="w-5 h-5 text-moss-600 stroke-[1.5]" />
                Diet
              </h3>
              <p className="text-sm font-light leading-relaxed text-bark-700">
                {diet || 'Information pending details.'}
              </p>
            </div>
          </div>

          {/* Behaviour & Vocalization */}
          <div className="space-y-3 border-t border-bark-200/50 pt-8">
            <h3 className="font-serif text-lg font-bold text-bark-900 flex items-center gap-2">
              <Info className="w-5 h-5 text-moss-600 stroke-[1.5]" />
              Natural Behaviour
            </h3>
            <p className="text-sm font-light leading-relaxed text-bark-700">
              {behaviour}
            </p>
            {vocalization && (
              <p className="text-sm leading-relaxed text-bark-600 italic bg-ivory-200/40 p-3.5 rounded-xl border border-ivory-300/30">
                <strong className="not-italic uppercase text-xs font-mono tracking-wider text-bark-600 block mb-1">Vocalizations</strong>
                "{vocalization}"
              </p>
            )}
          </div>

          {/* Breeding & Ecological Role */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 border-t border-bark-200/50 pt-8">
            <div className="space-y-2">
              <span className="text-xs font-mono uppercase tracking-widest text-moss-600 font-semibold block">Nesting & Breeding</span>
              <p className="text-sm font-light leading-relaxed text-bark-700">
                {breeding || 'Information pending details.'}
              </p>
            </div>
            
            <div className="space-y-2">
              <span className="text-xs font-mono uppercase tracking-widest text-moss-600 font-semibold block">Ecological Importance</span>
              <p className="text-sm font-light leading-relaxed text-bark-700">
                {ecological_importance || 'Plays a standard regulatory role in local insect and seed populations.'}
              </p>
            </div>
          </div>

        </div>
      </div>

      {/* Interesting Facts & Details */}
      {interesting_facts && interesting_facts.length > 0 && (
        <div className="border-t border-bark-200 pt-10 space-y-6">
          <h3 className="font-serif text-2xl text-bark-900 font-semibold text-center md:text-left">
            Observations & Interesting Facts
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6">
            {interesting_facts.map((fact, idx) => (
              <div 
                key={idx} 
                className="bg-ivory-200/30 hover:bg-ivory-200/60 p-4 rounded-xl border border-ivory-300/20 transition-colors flex items-start space-x-3 text-sm text-bark-700 font-light"
              >
                <div className="w-5 h-5 rounded-full bg-moss-600 text-white font-mono text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                  {idx + 1}
                </div>
                <p className="leading-relaxed">{fact}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Decorative Naturalist Sketch Placeholder */}
      <div className="flex justify-center border-t border-bark-200/50 pt-8 opacity-40">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 30" className="w-48 stroke-bark-600 fill-none stroke-[0.5]">
          <path d="M10 15 c20 -10, 40 10, 80 0" strokeDasharray="2,2" />
          <path d="M30 10 c10 -15, 20 15, 40 0" />
          <circle cx="50" cy="12" r="1" className="fill-bark-600" />
        </svg>
      </div>

    </motion.div>
  )
}

// Helper to remove index prefix from common name if displayed, e.g. "017.Cardinal" -> "Cardinal"
function commonNameCleanup(name: string): string {
  if (!name) return ''
  if (name.includes('.')) {
    name = name.split('.', 2)[1]
  }
  return name.replace(/_/g, ' ')
}

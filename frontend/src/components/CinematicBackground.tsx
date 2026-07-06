import React from 'react'

export default function CinematicBackground() {
  return (
    <div className="absolute inset-0 w-full h-full overflow-hidden select-none pointer-events-none bg-gradient-to-b from-forest-950 via-forest-900 to-forest-950 z-0">
      
      {/* Dynamic mist/fog svg overlay */}
      <div className="absolute inset-0 opacity-15 mix-blend-screen pointer-events-none">
        <svg viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
          <filter id="mist-filter">
            <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="3" result="noise" />
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="50" xChannelSelector="R" yChannelSelector="G" />
          </filter>
          <rect width="100%" height="100%" filter="url(#mist-filter)" className="fill-forest-400" />
        </svg>
      </div>

      {/* Swaying canopy layer */}
      <div className="absolute inset-x-0 top-0 h-64 md:h-96 opacity-10 pointer-events-none">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full fill-moss-400 wind-sway">
          <path d="M0,0 L100,0 L100,10 C80,20 60,5 50,15 C30,25 10,12 0,22 Z" />
        </svg>
      </div>

      {/* Forest base silhouette at the bottom */}
      <div className="absolute inset-x-0 bottom-0 h-48 md:h-72 opacity-20 pointer-events-none">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full fill-forest-900">
          <path d="M0,100 L100,100 L100,60 C90,65 80,55 70,62 C50,48 40,70 30,58 C15,68 8,50 0,65 Z" />
        </svg>
      </div>

      {/* Floating leaves particles (gentle animation) */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-1">
        <div className="absolute top-[20%] left-[15%] w-2 h-4 bg-moss-500/20 rounded-full rotate-45 animate-float-slow" style={{ animationDuration: '8s' }}></div>
        <div className="absolute top-[60%] left-[75%] w-3 h-5 bg-moss-600/10 rounded-full -rotate-12 animate-float-slow" style={{ animationDuration: '12s', animationDelay: '2s' }}></div>
        <div className="absolute top-[40%] left-[45%] w-1.5 h-3.5 bg-forest-400/20 rounded-full rotate-12 animate-float-slow" style={{ animationDuration: '10s', animationDelay: '4s' }}></div>
      </div>

      {/* Soft overlay gradient to ensure readability */}
      <div className="absolute inset-0 bg-gradient-to-t from-forest-950 via-transparent to-forest-950/40 pointer-events-none z-0"></div>
    </div>
  )
}

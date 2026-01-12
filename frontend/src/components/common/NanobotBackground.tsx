import { useEffect, useRef } from 'react'

interface Particle {
  x: number
  y: number
  size: number
  speedX: number
  speedY: number
  opacity: number
  hue: number
}

interface NanobotBackgroundProps {
  particleCount?: number
  connectionDistance?: number
  className?: string
}

export default function NanobotBackground({
  particleCount = 80,
  connectionDistance = 120,
  className = ''
}: NanobotBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const particlesRef = useRef<Particle[]>([])
  const mouseRef = useRef({ x: -1000, y: -1000 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // Track mouse position
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', handleMouseMove)

    // Initialize particles
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 2 + 1,
      speedX: (Math.random() - 0.5) * 0.5,
      speedY: (Math.random() - 0.5) * 0.5,
      opacity: Math.random() * 0.5 + 0.2,
      hue: Math.random() > 0.7 ? 280 : 190 // Cyan or purple
    }))

    // Animation loop
    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 15, 0.15)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const particles = particlesRef.current
      const mouse = mouseRef.current

      particles.forEach((p, i) => {
        // Mouse interaction - particles are attracted slightly
        const dx = mouse.x - p.x
        const dy = mouse.y - p.y
        const distToMouse = Math.sqrt(dx * dx + dy * dy)

        if (distToMouse < 200) {
          const force = (200 - distToMouse) / 200 * 0.02
          p.speedX += dx * force * 0.01
          p.speedY += dy * force * 0.01
        }

        // Apply velocity with damping
        p.x += p.speedX
        p.y += p.speedY
        p.speedX *= 0.99
        p.speedY *= 0.99

        // Add base velocity if too slow
        if (Math.abs(p.speedX) < 0.1) p.speedX += (Math.random() - 0.5) * 0.1
        if (Math.abs(p.speedY) < 0.1) p.speedY += (Math.random() - 0.5) * 0.1

        // Wrap around edges
        if (p.x < 0) p.x = canvas.width
        if (p.x > canvas.width) p.x = 0
        if (p.y < 0) p.y = canvas.height
        if (p.y > canvas.height) p.y = 0

        // Draw connections to nearby particles
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j]
          const dist = Math.sqrt((p.x - p2.x) ** 2 + (p.y - p2.y) ** 2)

          if (dist < connectionDistance) {
            const alpha = (1 - dist / connectionDistance) * 0.15
            const gradient = ctx.createLinearGradient(p.x, p.y, p2.x, p2.y)
            gradient.addColorStop(0, `hsla(${p.hue}, 100%, 60%, ${alpha})`)
            gradient.addColorStop(1, `hsla(${p2.hue}, 100%, 60%, ${alpha})`)

            ctx.beginPath()
            ctx.moveTo(p.x, p.y)
            ctx.lineTo(p2.x, p2.y)
            ctx.strokeStyle = gradient
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }

        // Draw particle with glow
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)

        // Outer glow
        const glowGradient = ctx.createRadialGradient(
          p.x, p.y, 0,
          p.x, p.y, p.size * 4
        )
        glowGradient.addColorStop(0, `hsla(${p.hue}, 100%, 60%, ${p.opacity})`)
        glowGradient.addColorStop(1, `hsla(${p.hue}, 100%, 60%, 0)`)
        ctx.fillStyle = glowGradient
        ctx.fill()

        // Core
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size * 0.5, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${p.hue}, 100%, 80%, ${p.opacity * 1.5})`
        ctx.fill()
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('mousemove', handleMouseMove)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [particleCount, connectionDistance])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: -1
      }}
    />
  )
}

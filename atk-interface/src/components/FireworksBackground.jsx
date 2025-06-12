import React, {
    useState,
    useEffect,
    useRef,
    useMemo,
    useCallback,
  } from 'react';
  import { motion } from 'framer-motion';
  
  /* ---------- helpers ---------- */
  
  const random = (min, max) => Math.random() * (max - min) + min;
  const uuid = () =>
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  
// Define possible firework colors
const fireworkColors = [
//   '#FF5733', // Orange-Red. NOTE we'll use a round of red fireworks for deletion and a round of bright green ones for downloading. Also we want the download/delete buttons to follow along the side of the page when scrolling far down due to a lot of items -- I Will make that change in a bit.
//   '#FFC300', // Vivid Yellow
//   '#DAF7A6', // Light Green
//   '#A0E7E5', // Light Blue
//   '#B4A0E7', // Lavender
//   '#E7A0C4', // Pink
  '#FFFFFF', // White
];
/* =================================================================== */
/*                               MAIN                                  */
/* =================================================================== */

const FireworksBackground = ({
  launchInterval = 5000,
  triggerBurst = false,
  burstColorOverride = null,
}) => {
    /* ---------- size ---------- */
  
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const containerRef = useRef(null);
  
    useEffect(() => {
      const handleResize = () => {
        if (containerRef.current) {
          setDimensions({
            width: containerRef.current.offsetWidth,
            height: containerRef.current.offsetHeight,
          });
        }
      };
      handleResize();
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }, []);
  
    /* ---------- pre-computed launch paths ---------- */
  
    const numPaths = 7;
    const paths = useMemo(() => {
      const { width, height } = dimensions;
      if (!width || !height) return [];
  
      return Array.from({ length: numPaths }, (_, id) => ({
        id,
        startX: width / 2,
        startY: height,
        endX: random(width * 0.2, width * 0.8),
        endY: random(height * 0.1, height * 0.4),
        launchDuration: random(0.8, 3), // s
        explosionDuration: random(1, 5), // s
        numParticles: Math.floor(random(15, 30)),
        numRings: Math.floor(random(2, 7)),
        particleRadius: random(80, 180),
        defaultColor:
          fireworkColors[Math.floor(random(0, fireworkColors.length))],
      }));
    }, [dimensions]);
  
    /* ---------- bookkeeping ---------- */
  
    const [activeFireworks, setActiveFireworks] = useState([]); // {key, pathId, color}
    const activePathIds = useRef(new Set());
  
    const removeFirework = useCallback((key, pathId) => {
      setActiveFireworks((prev) => prev.filter((fw) => fw.key !== key));
      activePathIds.current.delete(pathId);
    }, []);
  
    /* ---------- launch logic ---------- */
  
    const launchFirework = useCallback(
      (specificPathId = null, overrideColor = null) => {
        if (!paths.length) return;
  
        let pathId = specificPathId;
        if (pathId == null) {
          const available = paths
            .map((p) => p.id)
            .filter((id) => !activePathIds.current.has(id));
          if (!available.length) return; // all busy
          pathId = available[Math.floor(random(0, available.length))];
        }
  
        const path = paths.find(p => p.id === pathId);
        if (!path || activePathIds.current.has(pathId)) return; // already occupied or path not found
  
        const key = uuid();
        const color = overrideColor || path.defaultColor;
  
        activePathIds.current.add(pathId);
        setActiveFireworks((prev) => [...prev, { key, pathId, color }]);
      },
      [paths]
    );
  
    /* interval launch */
    const intervalRef = useRef();
    useEffect(() => {
      clearInterval(intervalRef.current);
      if (!paths.length) return;
  
      intervalRef.current = setInterval(() => launchFirework(null, null), launchInterval);
      return () => clearInterval(intervalRef.current);
    }, [launchInterval, launchFirework, paths.length]);
  
    /* burst launch */
    useEffect(() => {
      if (triggerBurst && paths.length) {
        paths.forEach((p, i) =>
          setTimeout(() => launchFirework(p.id, burstColorOverride), i * 50)
        );
      }
    }, [triggerBurst, burstColorOverride, paths, launchFirework]);
  
    /* ================================================================= */
    /*                              render                               */
    /* ================================================================= */
  
    return (
      <div
        ref={containerRef}
        className="absolute inset-0 overflow-hidden z-0 pointer-events-none"
        style={{ backgroundColor: 'rgba(17,24,39,0.7)' }}
      >
        {dimensions.width > 0 && (
          <svg width="100%" height="100%" style={{ position: 'absolute' }}>
            {activeFireworks.map(({ key, pathId, color }) => {
              const path = paths.find((p) => p.id === pathId);
              if (!path) return null;
              return (
                <Firework
                  key={key}
                  path={path}
                  color={color}
                  onDone={() => removeFirework(key, pathId)}
                />
              );
            })}
          </svg>
        )}
      </div>
    );
  };
  
  export default FireworksBackground;
  
  /* =================================================================== */
  /*                          Firework (declarative)                     */
  /* =================================================================== */
  
  const Firework = ({ path, color, onDone }) => {
    const [phase, setPhase] = useState('launch'); // launch → explode → done
  
    /* phase timers */
    useEffect(() => {
      const t1 = setTimeout(
        () => setPhase('explode'),
        path.launchDuration * 1000
      );
      const t2 = setTimeout(
        () => setPhase('done'),
        (path.launchDuration + path.explosionDuration) * 1000
      );
      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
      };
    }, [path]);
  
    /* notify parent */
    useEffect(() => {
      if (phase === 'done') onDone();
    }, [phase, onDone]);
  
    /* launch particle style */
    const launchMotion = {
      translateX: path.endX - path.startX,
      translateY: path.endY - path.startY,
      opacity: 0,
    };
  
    return (
      <g>
        {/* launch particle */}
        <motion.circle
          cx={path.startX}
          cy={path.startY}
          r={3}
          fill={color}
          initial={{ opacity: 1 }}
          animate={phase === 'launch' ? launchMotion : { opacity: 0 }}
          transition={{ duration: path.launchDuration, ease: 'linear' }}
        />
  
        {/* explosion group */}
        <motion.g
          style={{ x: path.endX, y: path.endY }}
          initial={{ opacity: 0 }}
          animate={{ opacity: phase === 'explode' ? 1 : 0 }}
        >
          {/* Create multiple rings for a fuller explosion */}
          {Array.from({ length: path.numRings }).flatMap((_, ringIndex) => {
            const ringDelay = ringIndex * 0.1; // Delay subsequent rings
            const ringRadiusMultiplier = 1 - ringIndex * 0.2; // Inner rings are slightly smaller (optional)

            return Array.from({ length: path.numParticles }).map((_, particleIndex) => {
              const angle = (particleIndex / path.numParticles) * 2 * Math.PI;
              const particleRadius = path.particleRadius * ringRadiusMultiplier;

              return (
                <motion.circle
                  // Need a unique key combining ring and particle index
                  key={`${ringIndex}-${particleIndex}`}
                  cx={0}
                  cy={0}
                  r={2.5}
                  fill={color}
                  initial={{ opacity: 1 }}
                  animate={
                    phase === 'explode'
                      ? {
                          translateX: Math.cos(angle) * particleRadius,
                          translateY: Math.sin(angle) * particleRadius,
                          opacity: 0, // fade out while travelling
                        }
                      : { opacity: 1 } // invisible anyway because group opacity = 0
                  }
                  transition={{
                    duration: path.explosionDuration,
                    ease: 'easeOut',
                    // Combine ring delay with random individual delay
                    delay: ringDelay + Math.random() * 0.15,
                  }}
                />
              );
            });
          })}
        </motion.g>
      </g>
    );
  };
import React, { useMemo, useEffect, useRef } from 'react';
import { motion, useMotionValue, useTransform, useAnimationFrame } from 'framer-motion';

// Function to generate a random number within a range
const random = (min, max) => Math.random() * (max - min) + min;

// Create a single line element
const Line = ({ width, delay }) => (
  <motion.div
    className="h-1.5 bg-gray-600 rounded"
    style={{ width: `${width}%` }}
    initial={{ opacity: 0 }}
    animate={{ opacity: 0.4 }}
    transition={{ delay, duration: 0.5 }}
  />
);

// Create the scrolling background component
const ScrollingTextBackground = ({ duration = 30 }) => {
  const numLines = 50;

  const lines = useMemo(() => {
    return Array.from({ length: numLines }).map((_, i) => ({
      id: i,
      width: random(20, 80),
      delay: random(0, 1),
    }));
  }, []);

  // --- Manual Animation Logic ---
  const yProgress = useMotionValue(0); // Tracks progress from 0 to 1
  const yPercentage = useTransform(yProgress, [0, 1], ["0%", "-50%"]); // Maps progress to Y offset

  const lastFrameTime = useRef(performance.now());
  const animationFrameId = useRef(null); // To store the request ID
  const currentDuration = useRef(duration); // Store duration in a ref

  // Update the duration ref whenever the prop changes
  useEffect(() => {
    currentDuration.current = duration;
  }, [duration]);

  useAnimationFrame((timestamp) => {
    const deltaTime = timestamp - lastFrameTime.current;

    // Guard against potentially huge deltaTime after remount or tab backgrounding.
    // Only update animation if deltaTime is reasonable (e.g., < 1 second).
    // Also handle potential negative deltaTime if clocks adjust.
    if (deltaTime < 0 || deltaTime > 1000) {
        lastFrameTime.current = timestamp; // Reset for the next frame
        return; // Skip animation update for this frame
    }

    lastFrameTime.current = timestamp; // Update time for the next frame calculation

    // Calculate progress increment based on current duration
    // duration is in seconds, deltaTime in ms. Convert duration to ms.
    const progressIncrement = deltaTime / (currentDuration.current * 1000);

    // Update progress, wrapping around using modulo
    const newProgress = (yProgress.get() + progressIncrement) % 1;
    yProgress.set(newProgress);
  });

  // --- End Manual Animation Logic ---

  return (
    <div className="absolute inset-0 overflow-hidden z-0 pointer-events-none">
      <motion.div
        className="w-full"
        style={{ y: yPercentage }} // Apply the transformed y value directly
        // No animate, variants, or initial props needed for y animation here
      >
        {/* First block of lines */}
        <div className="w-full flex flex-col items-center space-y-4">
          {lines.map(line => (
            <Line key={`top-${line.id}`} width={line.width} delay={line.delay} />
          ))}
        </div>
        <br />
        {/* Second block of lines */}
        <div className="w-full flex flex-col items-center space-y-4">
          {lines.map(line => (
            <Line key={`bottom-${line.id}`} width={line.width} delay={line.delay} />
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default ScrollingTextBackground; 
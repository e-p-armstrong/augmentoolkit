import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const random = (min, max) => Math.random() * (max - min) + min;

const Particle = ({ id, color, size, duration }) => {
  const angle = random(0, Math.PI * 2);
  const radius = random(40, 85); // How far particles travel (Increased)
  const endX = Math.cos(angle) * radius;
  const endY = Math.sin(angle) * radius;

  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        backgroundColor: color,
        width: size,
        height: size,
        left: '50%', // Start at center
        top: '50%',
        translateX: '-50%', // Adjust for centering
        translateY: '-50%',
      }}
      initial={{ opacity: 1, scale: 1 }}
      animate={{
        x: endX,
        y: endY,
        scale: 0, // Shrink to disappear
        opacity: 0,
      }}
      transition={{ duration: duration, ease: 'easeOut' }}
    />
  );
};

const ButtonParticles = ({ trigger, color = '#ffffff', count = 8, size = 3.5, duration = 0.45 }) => {
  const [particles, setParticles] = useState([]);

  console.log(`ButtonParticles: Rendered. Current trigger=${trigger}, particles=${particles.length}`);

  useEffect(() => {
    console.log(`ButtonParticles: useEffect triggered. trigger=${trigger}`);
    if (trigger > 0) { // trigger increments on each click
      console.log(`ButtonParticles: Trigger > 0 (${trigger}). Generating ${count} particles.`);
      const newParticles = Array.from({ length: count }).map((_, i) => ({
        id: `${trigger}-${i}`, // Unique key based on trigger and index
        color,
        size,
        duration,
      }));
      setParticles(newParticles);
      console.log("ButtonParticles: Particles state updated:", newParticles);

      // Optional: Clear particles after animation - not strictly needed as they animate to opacity 0
      const timer = setTimeout(() => {
        console.log(`ButtonParticles: Clearing particles after ${duration}s timeout.`);
        setParticles([]);
      }, duration * 1000); // Clear particles after animation
      return () => {
        console.log("ButtonParticles: useEffect cleanup running.");
        clearTimeout(timer)
      };
    }
  }, [trigger, color, count, size, duration]);

  // Render only the particles currently animating
  return (
    <>
      {particles.map(p => (
        <React.Fragment key={p.id}> {/* Use Fragment to add log without extra div */}
          {/* {console.log(`ButtonParticles: Rendering Particle id=${p.id}`)} */}
          <Particle key={p.id} {...p} />
        </React.Fragment>
      ))}
    </>
  );
};

export default ButtonParticles; 
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, useAnimation } from 'framer-motion';

// --- Helper Functions ---
const random = (min, max) => Math.random() * (max - min) + min;
const uuid = () =>
    typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

// Interpolate between two colors
const interpolateColor = (color1, color2, factor) => {
    if (factor < 0) factor = 0;
    if (factor > 1) factor = 1;

    const c1 = parseInt(color1.substring(1), 16);
    const c2 = parseInt(color2.substring(1), 16);

    const r1 = (c1 >> 16) & 0xff;
    const g1 = (c1 >> 8) & 0xff;
    const b1 = c1 & 0xff;

    const r2 = (c2 >> 16) & 0xff;
    const g2 = (c2 >> 8) & 0xff;
    const b2 = c2 & 0xff;

    const r = Math.round(r1 + (r2 - r1) * factor);
    const g = Math.round(g1 + (g2 - g1) * factor);
    const b = Math.round(b1 + (b2 - b1) * factor);

    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase()}`;
};

const START_COLOR = '#4ade80'; // Dull Green (Tailwind green-400)
const END_COLOR = '#00FF00';   // Bright Neon Green

// --- Particle Component ---
const Particle = ({ id, initialX, initialY, targetPointX, targetPointY, color, status, screenHeight, screenWidth, onComplete }) => {
    const controls = useAnimation();
    const [currentState, setCurrentState] = useState(status);
    const particleRef = useRef({ isComplete: false }); // Ref to track completion status

    useEffect(() => {
        setCurrentState(status); // Update state if prop changes
    }, [status]);

    useEffect(() => {
        if (particleRef.current.isComplete) return; // Don't restart animation if already completing

        let animationSequence;
        const durationBase = random(6, 12); // Base duration for normal movement
        const fadeDuration = 1.5; // Normal fade duration

        // Slow fade durations for lingering particles
        const lingerFadeDuration = random(8, 15); // Longer fade for failed/revoked
        const lingerDelay = random(3, 7); // Delay before slow fade starts

        switch (currentState) {
            case 'FAILED':
                // Drop to bottom, linger as smaller "corpse", then fade slowly
                animationSequence = async () => {
                    // 1. Drop to near the bottom
                    await controls.start({
                        x: random(0, screenWidth), // Scatter slightly at the bottom
                        y: screenHeight - random(10, 15), // Settle very close to the bottom edge
                        opacity: 1,
                        scale: 0.8, // Slightly smaller "corpse"
                        transition: { duration: random(1, 2.5), ease: "easeIn" }
                    });
                    // 2. Linger, then fade slowly
                    await controls.start({
                        opacity: 0,
                        transition: { duration: lingerFadeDuration, delay: lingerDelay }
                    });
                    if (!particleRef.current.isComplete) {
                       particleRef.current.isComplete = true;
                       onComplete(id);
                    }
                };
                break;

            case 'REVOKED':
                 // Flee to edges, linger smaller/huddled, then fade slowly
                 {
                     const edgeX = initialX < screenWidth / 2 ? random(-15, 0) : random(screenWidth, screenWidth + 15);
                     const edgeY = random(0, screenHeight);
                     animationSequence = async () => {
                         // 1. Move quickly to the edge
                         await controls.start({
                             x: edgeX,
                             y: edgeY,
                             opacity: 1,
                             scale: 0.6, // Smaller, huddled look
                             transition: { duration: random(0.5, 1.5), ease: "easeOut" }
                         });
                         // 2. Linger, then fade slowly
                         await controls.start({
                             opacity: 0,
                             transition: { duration: lingerFadeDuration, delay: lingerDelay }
                         });
                         if (!particleRef.current.isComplete) {
                             particleRef.current.isComplete = true;
                             onComplete(id);
                         }
                     };
                 }
                 break;

            case 'RUNNING':
            case 'PENDING':
            case 'COMPLETED':
            default:
                // Move towards the target point on the bar and fade (original behavior)
                animationSequence = async () => {
                    // 1. Quick fade-in
                    await controls.start({
                        opacity: 1,
                        scale: 1, // Start at full size immediately
                        transition: { duration: 0.3, ease: "easeOut" } // Fast fade-in
                    });

                    // 2. Move towards the target point
                    await controls.start({
                        x: targetPointX,
                        y: targetPointY,
                        // Opacity and scale are already 1 from the previous step
                        transition: { duration: durationBase, ease: "circOut" }
                    });

                    // 3. Fade out near the target
                    await controls.start({
                        opacity: 0,
                        scale: 0.3, // Shrink slightly as it fades
                        transition: { duration: fadeDuration }
                    });
                    if (!particleRef.current.isComplete) {
                        particleRef.current.isComplete = true;
                        onComplete(id);
                    }
                };
                break;
        }

        animationSequence().catch(err => {
            // Handle potential animation errors (e.g., component unmounted)
            if (err.name !== 'AbortError' && !particleRef.current.isComplete) {
                console.error("Particle animation error:", err);
                 // Ensure cleanup happens even on error if not already completed
                 particleRef.current.isComplete = true;
                 onComplete(id);
            }
        });

        // Cleanup function for when component unmounts or dependencies change
        // This helps prevent calling onComplete multiple times if status changes rapidly
        return () => {
            controls.stop(); // Stop any ongoing animation
            // If the component unmounts before the animation naturally completes,
            // ensure we call onComplete if it hasn't been called yet.
            // This is important for states like RUNNING->FAILED transition.
             if (!particleRef.current.isComplete) {
                 // We might not call onComplete here directly,
                 // as the state change might trigger a new animation
                 // that WILL eventually call onComplete. Let the new effect handle it.
                 // If we *do* call it here, it might remove the particle before the
                 // FAILED/REVOKED animation can even start.
                 // Let's rely on the isComplete flag within the async sequences.
             }
        };

    // Ensure dependencies cover all inputs for the animation logic
    }, [controls, id, initialX, initialY, targetPointX, targetPointY, color, currentState, screenHeight, screenWidth, onComplete]);


    return (
        <motion.div
            key={id}
            // Make particles smaller
            className="absolute w-1.5 h-1.5 rounded-full pointer-events-none" // Size reduced
            style={{
                backgroundColor: color,
                // Slightly reduced glow for density
                boxShadow: `0 0 4px ${color}, 0 0 8px ${color}`, // Adjusted glow
                top: 0,
                left: 0,
            }}
            initial={{ x: initialX, y: initialY, opacity: 0, scale: 0.5 }}
            animate={controls}
        />
    );
};

// --- Main Particle Container ---
const ProgressBarParticles = ({ taskId, progress, status, progressBarElement, downloadButtonElement, color }) => {
    const [particles, setParticles] = useState([]);
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [barRect, setBarRect] = useState(null);
    const [buttonRect, setButtonRect] = useState(null); // State for button position

    // Get container dimensions
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.offsetWidth,
                    height: containerRef.current.offsetHeight,
                });
            }
        };
        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Get progress bar dimensions/position relative to the viewport
    useEffect(() => {
        const updateBarRect = () => {
            if (progressBarElement) {
                setBarRect(progressBarElement.getBoundingClientRect());
            } else {
                setBarRect(null);
            }
        };

        updateBarRect(); // Initial read

        // Use ResizeObserver for efficiency if needed, or just poll on interval/scroll
        const intervalId = setInterval(updateBarRect, 200); // Check position periodically

        window.addEventListener('scroll', updateBarRect); // Update on scroll

        return () => {
            clearInterval(intervalId);
            window.removeEventListener('scroll', updateBarRect);
        };
    }, [progressBarElement]); // Re-run if the element itself changes

    // Get download button dimensions/position relative to the viewport
    useEffect(() => {
        const updateButtonRect = () => {
            if (downloadButtonElement) {
                setButtonRect(downloadButtonElement.getBoundingClientRect());
            } else {
                setButtonRect(null);
            }
        };

        updateButtonRect(); // Initial read

        // Use ResizeObserver for efficiency if needed, or just poll on interval/scroll
        const intervalId = setInterval(updateButtonRect, 200); // Check position periodically

        window.addEventListener('scroll', updateButtonRect); // Update on scroll

        return () => {
            clearInterval(intervalId);
            window.removeEventListener('scroll', updateButtonRect);
        };
    }, [downloadButtonElement]); // Re-run if the button element itself changes

    const removeParticle = useCallback((id) => {
        setParticles(prev => prev.filter(p => p.id !== id));
    }, []);

    // Particle Generation Logic
    useEffect(() => {
        if (!taskId || !dimensions.width || !dimensions.height || !barRect) {
            if (!taskId && particles.length > 0) {
                setParticles([]);
            }
            return;
        }

        // --- Update existing particles status ---
        // Do this regardless of the current task status, so particles
        // correctly transition to FAILED/REVOKED animations etc.
        setParticles(prev => prev.map(p => p.status !== status ? { ...p, status: status } : p));

        // --- Particle Generation ---
        // Always generate particles, but their behavior is determined by the status prop passed to Particle

        // Increase max particles significantly & faster generation rate
        const maxParticles = 15 + Math.floor((progress ?? 0) * 150); // Halved: Max ~165 particles at 100%
        // Faster base rate, scales more aggressively with progress. Min 20ms interval.
        const generationRate = Math.max(40, 200 - (progress ?? 0) * 160); // Slower generation: interval 40ms (100%) to 200ms (0%)


        const intervalId = setInterval(() => {
            setParticles(prev => {
                // Check against maxParticles (use the length *after* potential updates)
                if (prev.length < maxParticles) {
                    const edge = Math.floor(random(0, 4)); // 0: top, 1: right, 2: bottom, 3: left
                    let startX, startY;
                    switch (edge) {
                        case 0: // top
                            startX = random(0, dimensions.width);
                            startY = -10;
                            break;
                        case 1: // right
                            startX = dimensions.width + 10;
                            startY = random(0, dimensions.height);
                            break;
                        case 2: // bottom
                            startX = random(0, dimensions.width);
                            startY = dimensions.height + 10;
                            break;
                        case 3: // left
                        default:
                            startX = -10;
                            startY = random(0, dimensions.height);
                            break;
                    }

                    // Calculate target point within the *filled* portion of the bar OR the button
                    const containerRect = containerRef.current?.getBoundingClientRect();
                    if (!containerRect) return prev; // Exit if container not ready

                    let targetX, targetY;

                    // Target button if status is COMPLETED and button exists
                    if (status === 'COMPLETED' && buttonRect) {
                        const buttonStartX = buttonRect.left - containerRect.left;
                        const buttonStartY = buttonRect.top - containerRect.top;
                        targetX = buttonStartX + random(0, buttonRect.width);
                        targetY = buttonStartY + random(0, buttonRect.height);
                    }
                    // Otherwise, target the progress bar (if it exists)
                    else if (barRect) {
                        const barStartX = barRect.left - containerRect.left;
                        const barStartY = barRect.top - containerRect.top;
                        const filledWidth = barRect.width * (progress ?? 0);

                        // Target a random X within the filled area, or close to start if progress is 0
                        targetX = barStartX + random(0, Math.max(5, filledWidth)); // Ensure minimum target width
                        // Target Y somewhere vertically within the bar height
                        targetY = barStartY + random(0, barRect.height);
                    }
                    // Fallback if neither is ready (should be rare)
                    else {
                        targetX = startX; // Stay near start position
                        targetY = startY;
                    }

                    return [
                        ...prev, // Use the already updated array if status changed, otherwise original prev
                        {
                            id: uuid(),
                            initialX: startX,
                            initialY: startY,
                            targetPointX: targetX,
                            targetPointY: targetY,
                            color: color,
                            status: status, // Pass current task status
                        }
                    ];
                }
                // If max particles reached, just return the potentially status-updated array
                return prev;
            });
        }, generationRate);

        return () => clearInterval(intervalId);
    }, [taskId, progress, status, dimensions, barRect, buttonRect, downloadButtonElement, removeParticle, color]);

    return (
        <div
            ref={containerRef}
            className="absolute inset-0 overflow-hidden z-0 pointer-events-none" // z-0 to be behind content
        >
            {particles.map(p => (
                <Particle
                    key={p.id}
                    {...p}
                    screenHeight={dimensions.height}
                    screenWidth={dimensions.width}
                    onComplete={removeParticle}
                />
            ))}
        </div>
    );
};

export default ProgressBarParticles; 
import React, { useState } from "react";
import { motion, useAnimation } from "framer-motion";
import ButtonParticles from "./ButtonParticles";

const EnhancedButton = ({
  onClick,
  children,
  className = "",
  disabled = false,
  title = "",
  particleColor = "#ffffff",
  particleCount = 50,
  particleSize = 10,
  particleDuration = 3,
  ...props
}) => {
  const [isHovering, setIsHovering] = useState(false);
  const [hasBeenClickedSinceHover, setHasBeenClickedSinceHover] = useState(false);
  const [particleTrigger, setParticleTrigger] = useState(0);

  /* --- controller that drives ONLY the shake --- */
  const shake = useAnimation();

  const fireShake = () =>
    shake.start({
      x: [0, -3, 3, -3, 3, 0],
      y: [0,  3, -3,  3, -3, 0],
      transition: { duration: 0.25, ease: "easeInOut" }
    });

  const handleClick = (e) => {
    if (disabled) return;
    onClick?.(e);
    setHasBeenClickedSinceHover(true);
    setParticleTrigger((t) => t + 1);
    fireShake();
  };

  /* -------- hover / normal variants (scale only) -------- */
  const variants = {
    initial:        { scale: 1, filter: "brightness(1)" },
    hoverInitial:   {
      scale: [1, 1.06, 1],
      transition: { scale: { duration: 0.8, repeat: Infinity, ease: "easeInOut" } }
    },
    hoverClicked:   { scale: 1.03, transition: { duration: 0.2 } }
  };

  const state =
    disabled
      ? "initial"
      : isHovering
      ? hasBeenClickedSinceHover
        ? "hoverClicked"
        : "hoverInitial"
      : "initial";

  /* ---------------- render ---------------- */
  return (
    /* wrapper receives the shake controller */
    <motion.div animate={shake} initial={{ x: 0, y: 0 }}>
      <motion.button
        type="button"
        className={`relative overflow-visible transition-colors duration-200 ease-in-out
                    ${className} ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
        title={title}
        disabled={disabled}
        variants={variants}
        initial="initial"
        animate={state}
        whileTap={!disabled ? { scale: 0.8 } : false}  /* press-in squash only */
        onHoverStart={() => { setIsHovering(true); setHasBeenClickedSinceHover(false); }}
        onHoverEnd={() => setIsHovering(false)}
        onClick={handleClick}
        {...props}
      >
        <span className="relative z-10 flex items-center justify-center gap-2">
          {children}
        </span>

        {/* particles */}
        <div className="absolute inset-0 z-0 pointer-events-none">
          <ButtonParticles
            trigger={particleTrigger}
            color={particleColor}
            count={particleCount}
            size={particleSize}
            duration={particleDuration}
          />
        </div>
      </motion.button>
    </motion.div>
  );
};

export default EnhancedButton;

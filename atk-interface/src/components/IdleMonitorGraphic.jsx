import React from 'react';
import { motion } from 'framer-motion';
import { FiSettings } from 'react-icons/fi'; // Keep gear

const IdleMonitorGraphic = () => {
  return (
    <div className="relative flex items-center justify-center w-40 h-40 text-gray-600"> {/* Container size (controls gear size) */}
      {/* Spinning Gear */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          repeat: Infinity,
          duration: 15, // Slower spin
          ease: 'linear',
        }}
        className="absolute inset-0 flex items-center justify-center"
      >
        {/* Gear takes color from parent div */}
        <FiSettings size={200} /> {/* Increased size */}
      </motion.div>

      {/* Stationary Binoculars SVG - Centered and Resized */}
      {/* Container centers the SVG within the gear */}
      <div className="absolute flex items-center justify-center text-gray-500"> 
         {/* SVG adjusted: size, viewBox might need tuning based on visual */}
         <svg 
           version="1.1" 
           id="Capa_1" 
           xmlns="http://www.w3.org/2000/svg" 
           xmlnsXlink="http://www.w3.org/1999/xlink" 
           width="65" /* Adjusted size */
           height="65" /* Adjusted size */
           viewBox="0 0 921.998 921.998" 
           xmlSpace="preserve"
           // Removed explicit fill from SVG tag
         >
           {/* Group takes fill from parent div's text color */}
           <g fill="currentColor"> 
             <g>
               {/* Path data remains the same */}
               <path d="M869.694,385.652c-11.246-12.453-132.373-110.907-154.023-117.272c-9.421-2.735-18.892-4.447-28.681-5.164
                 c-45.272-3.315-95.213,10.875-126.684,44.794c-2.741,2.956-4.311,4.645-4.311,4.645s1.172-1.996,3.224-5.488
                 c9.706-16.365,23.847-30.577,38.989-41.956c6.979-5.243,14.37-9.937,22.088-14.014c2.116-1.118,21.797-11.751,23.12-10.357
                 c-0.003-0.003-10.744-11.33-10.744-11.33c-17.273-17.276-35.963-32.167-61.415-32.167c-31.547,0-58.505,19.559-69.472,47.201
                 c-9.306-6.917-24.11-11.392-40.788-11.392c-16.678,0-31.481,4.475-40.788,11.392c-10.967-27.643-37.925-47.201-69.472-47.201
                 c-25.452,0-44.142,14.891-61.416,32.166c0,0-10.741,11.327-10.744,11.33c1.322-1.395,21.003,9.239,23.12,10.357
                 c7.718,4.077,15.109,8.771,22.088,14.014c15.145,11.378,29.283,25.591,38.989,41.956c2.052,3.493,3.224,5.488,3.224,5.488
                 s-1.566-1.689-4.31-4.645c-31.471-33.919-81.411-48.109-126.683-44.794c-9.789,0.717-19.26,2.429-28.681,5.164
                 c-21.651,6.365-142.778,104.819-154.023,117.272C19.797,421.645,0,469.336,0,521.655c0,112.112,90.886,203,203,203
                 c102.56,0,187.34-76.062,201.048-174.851c15.983,11.645,35.663,18.52,56.951,18.52c21.289,0,40.968-6.875,56.951-18.52
                 c13.708,98.788,98.487,174.851,201.048,174.851c112.114,0,203-90.888,203-203C921.996,469.336,902.199,421.647,869.694,385.652z
                 M198.497,649.155c-67.611,0-122.421-54.811-122.421-122.421s54.81-122.42,122.421-122.42s122.421,54.81,122.421,122.42
                 S266.108,649.155,198.497,649.155z M460.997,515.234c-17.833,0-32.29-14.457-32.29-32.29s14.457-32.289,32.29-32.289
                 s32.29,14.457,32.29,32.289C493.287,500.777,478.83,515.234,460.997,515.234z M723.497,649.155
                 c-67.611,0-122.421-54.811-122.421-122.421s54.81-122.42,122.421-122.42s122.421,54.81,122.421,122.42
                 S791.108,649.155,723.497,649.155z"/>
             </g>
           </g>
         </svg>
      </div>
    </div>
  );
};

export default IdleMonitorGraphic; 
import React, { useState, useCallback } from 'react';
import { StarIcon } from '@heroicons/react/24/solid';
import { StarIcon as StarOutlineIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface StarRatingProps {
  rating: number;
  maxRating?: number;
  size?: 'sm' | 'md' | 'lg';
  interactive?: boolean;
  showLabel?: boolean;
  clearable?: boolean;
  disabled?: boolean;
  onChange?: (rating: number) => void;
  onHover?: (rating: number) => void;
  className?: string;
}

export const StarRating: React.FC<StarRatingProps> = ({
  rating,
  maxRating = 5,
  size = 'md',
  interactive = false,
  showLabel = false,
  clearable = false,
  disabled = false,
  onChange,
  onHover,
  className,
}) => {
  const [hoverRating, setHoverRating] = useState<number>(0);
  const [isHovering, setIsHovering] = useState(false);
  
  const displayRating = isHovering ? hoverRating : rating;
  
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };
  
  const handleMouseEnter = useCallback((starRating: number) => {
    if (!interactive || disabled) return;
    
    setHoverRating(starRating);
    setIsHovering(true);
    onHover?.(starRating);
  }, [interactive, disabled, onHover]);
  
  const handleMouseLeave = useCallback(() => {
    if (!interactive || disabled) return;
    
    setIsHovering(false);
    setHoverRating(0);
    onHover?.(0);
  }, [interactive, disabled, onHover]);
  
  const handleClick = useCallback((starRating: number) => {
    if (!interactive || disabled) return;
    
    // If clearable and clicking on current rating, clear it
    if (clearable && starRating === rating) {
      onChange?.(0);
    } else {
      onChange?.(starRating);
    }
  }, [interactive, disabled, clearable, rating, onChange]);
  
  const handleKeyDown = useCallback((event: React.KeyboardEvent, starRating: number) => {
    if (!interactive || disabled) return;
    
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick(starRating);
    }
  }, [interactive, disabled, handleClick]);
  
  return (
    <div className={clsx('flex items-center gap-1', className)}>
      {/* Stars */}
      <div className="flex items-center">
        {Array.from({ length: maxRating }, (_, index) => {
          const starRating = index + 1;
          const isFilled = starRating <= displayRating;
          
          return (
            <button
              key={starRating}
              type="button"
              disabled={!interactive || disabled}
              className={clsx(
                sizeClasses[size],
                'transition-all duration-150 ease-in-out',
                {
                  // Interactive styles
                  'cursor-pointer hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1': 
                    interactive && !disabled,
                  'cursor-default': !interactive || disabled,
                  // Color styles
                  'text-rating-star': isFilled,
                  'text-rating-empty': !isFilled,
                  // Hover effects
                  'hover:text-rating-starHover': interactive && !disabled && !isFilled,
                  // Disabled styles
                  'opacity-50': disabled,
                }
              )}
              onMouseEnter={() => handleMouseEnter(starRating)}
              onMouseLeave={handleMouseLeave}
              onClick={() => handleClick(starRating)}
              onKeyDown={(e) => handleKeyDown(e, starRating)}
              aria-label={`Rate ${starRating} star${starRating > 1 ? 's' : ''}`}
              tabIndex={interactive && !disabled ? 0 : -1}
            >
              {isFilled ? (
                <StarIcon className="w-full h-full" />
              ) : (
                <StarOutlineIcon className="w-full h-full" />
              )}
            </button>
          );
        })}
      </div>
      
      {/* Rating label */}
      {showLabel && (
        <span className="text-sm text-secondary-600 ml-2">
          {rating > 0 ? `${rating}/${maxRating}` : 'Not rated'}
        </span>
      )}
      
      {/* Hover feedback */}
      {interactive && isHovering && hoverRating > 0 && (
        <span className="text-xs text-secondary-500 ml-2">
          {clearable && hoverRating === rating ? 'Click to clear' : `Rate ${hoverRating} star${hoverRating > 1 ? 's' : ''}`}
        </span>
      )}
    </div>
  );
};
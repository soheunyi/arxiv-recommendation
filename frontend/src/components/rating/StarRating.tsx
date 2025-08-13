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
  pendingRating?: number; // For optimistic updates
  isUpdating?: boolean; // To show loading state
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
  pendingRating,
  isUpdating = false,
  onChange,
  onHover,
  className,
}) => {
  const [hoverRating, setHoverRating] = useState<number>(0);
  const [isHovering, setIsHovering] = useState(false);
  
  // Use pending rating for optimistic updates, fallback to actual rating
  const currentRating = pendingRating !== undefined ? pendingRating : rating;
  const displayRating = isHovering ? hoverRating : currentRating;
  
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };
  
  const handleMouseEnter = useCallback((starRating: number) => {
    if (!interactive || disabled || isUpdating) return;
    
    setHoverRating(starRating);
    setIsHovering(true);
    onHover?.(starRating);
  }, [interactive, disabled, isUpdating, onHover]);
  
  const handleMouseLeave = useCallback(() => {
    if (!interactive || disabled || isUpdating) return;
    
    setIsHovering(false);
    setHoverRating(0);
    onHover?.(0);
  }, [interactive, disabled, isUpdating, onHover]);
  
  const handleClick = useCallback((starRating: number) => {
    if (!interactive || disabled || isUpdating) return;
    
    // If clearable and clicking on current rating, clear it
    if (clearable && starRating === currentRating) {
      onChange?.(0);
    } else {
      onChange?.(starRating);
    }
  }, [interactive, disabled, isUpdating, clearable, currentRating, onChange]);
  
  const handleKeyDown = useCallback((event: React.KeyboardEvent, starRating: number) => {
    if (!interactive || disabled || isUpdating) return;
    
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick(starRating);
    }
  }, [interactive, disabled, handleClick]);
  
  return (
    <div className={clsx('flex items-center gap-1', className)} data-testid="star-rating">
      {/* Stars */}
      <div className="flex items-center">
        {Array.from({ length: maxRating }, (_, index) => {
          const starRating = index + 1;
          const isFilled = starRating <= displayRating;
          
          return (
            <button
              key={starRating}
              type="button"
              disabled={!interactive || disabled || isUpdating}
              className={clsx(
                sizeClasses[size],
                'transition-all duration-150 ease-in-out',
                {
                  // Interactive styles
                  'cursor-pointer hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1': 
                    interactive && !disabled && !isUpdating,
                  'cursor-default': !interactive || disabled || isUpdating,
                  // Color styles
                  'text-rating-star': isFilled,
                  'text-rating-empty': !isFilled,
                  // Hover effects
                  'hover:text-rating-starHover': interactive && !disabled && !isUpdating && !isFilled,
                  // Disabled/updating styles
                  'opacity-50': disabled,
                  'opacity-75 animate-pulse': isUpdating,
                }
              )}
              onMouseEnter={() => handleMouseEnter(starRating)}
              onMouseLeave={handleMouseLeave}
              onClick={() => handleClick(starRating)}
              onKeyDown={(e) => handleKeyDown(e, starRating)}
              aria-label={`Rate ${starRating} star${starRating > 1 ? 's' : ''}`}
              tabIndex={interactive && !disabled && !isUpdating ? 0 : -1}
            >
              {isFilled ? (
                <StarIcon className="w-full h-full" data-testid="star-filled" />
              ) : (
                <StarOutlineIcon className="w-full h-full" data-testid="star-empty" />
              )}
            </button>
          );
        })}
      </div>
      
      {/* Rating label */}
      {showLabel && (
        <span className="text-sm text-secondary-600 ml-2">
          {currentRating > 0 ? `${currentRating}/${maxRating}` : 'Not rated'}
          {isUpdating && <span className="text-xs text-secondary-400 ml-1">(updating...)</span>}
        </span>
      )}
      
      {/* Hover feedback */}
      {interactive && isHovering && hoverRating > 0 && !isUpdating && (
        <span className="text-xs text-secondary-500 ml-2">
          {clearable && hoverRating === currentRating ? 'Click to clear' : `Rate ${hoverRating} star${hoverRating > 1 ? 's' : ''}`}
        </span>
      )}
    </div>
  );
};
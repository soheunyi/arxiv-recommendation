import React, { useState, useEffect } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import clsx from 'clsx';

interface AbstractRendererProps {
  abstract: string;
  expanded?: boolean;
  onToggle?: () => void;
  maxLength?: number;
  showToggle?: boolean;
  className?: string;
}

export const AbstractRenderer: React.FC<AbstractRendererProps> = ({
  abstract,
  expanded = false,
  onToggle,
  maxLength = 300,
  showToggle = true,
  className,
}) => {
  const [processedContent, setProcessedContent] = useState<React.ReactNode[]>([]);
  const [hasLatex, setHasLatex] = useState(false);
  
  useEffect(() => {
    if (!abstract) {
      setProcessedContent([]);
      setHasLatex(false);
      return;
    }
    
    try {
      const content = processLatexContent(abstract);
      setProcessedContent(content);
      setHasLatex(detectLatex(abstract));
    } catch (error) {
      console.warn('LaTeX processing failed, falling back to plain text:', error);
      setProcessedContent([abstract]);
      setHasLatex(false);
    }
  }, [abstract]);
  
  const detectLatex = (text: string): boolean => {
    const latexPatterns = [
      /\$\$[^$]+\$\$/g,  // Display math $$...$$
      /\$[^$\n]+\$/g,    // Inline math $...$
      /\\[a-zA-Z]+/g,    // LaTeX commands
    ];
    
    return latexPatterns.some(pattern => pattern.test(text));
  };
  
  const processLatexContent = (text: string): React.ReactNode[] => {
    const elements: React.ReactNode[] = [];
    let currentIndex = 0;
    let elementKey = 0;
    
    // Find all math expressions
    const mathExpressions: Array<{ start: number; end: number; type: 'block' | 'inline'; content: string }> = [];
    
    // Find display math $$...$$
    text.replace(/\$\$([^$]+?)\$\$/g, (match, content, offset) => {
      mathExpressions.push({
        start: offset,
        end: offset + match.length,
        type: 'block',
        content: content.trim(),
      });
      return match;
    });
    
    // Find inline math $...$
    text.replace(/\$([^$\n]+?)\$/g, (match, content, offset) => {
      // Skip if this is part of a display math expression
      const isInDisplayMath = mathExpressions.some(expr => 
        expr.type === 'block' && offset >= expr.start && offset < expr.end
      );
      
      if (!isInDisplayMath) {
        mathExpressions.push({
          start: offset,
          end: offset + match.length,
          type: 'inline',
          content: content.trim(),
        });
      }
      return match;
    });
    
    // Sort by position
    mathExpressions.sort((a, b) => a.start - b.start);
    
    // Process text with math expressions
    mathExpressions.forEach(expr => {
      // Add text before math expression
      if (currentIndex < expr.start) {
        const textBefore = text.slice(currentIndex, expr.start);
        if (textBefore) {
          elements.push(<span key={elementKey++}>{textBefore}</span>);
        }
      }
      
      // Add math expression
      try {
        if (expr.type === 'block') {
          elements.push(
            <div key={elementKey++} className="my-2">
              <BlockMath math={expr.content} />
            </div>
          );
        } else {
          elements.push(
            <span key={elementKey++}>
              <InlineMath math={expr.content} />
            </span>
          );
        }
      } catch (error) {
        // Fallback to original text if LaTeX rendering fails
        console.warn(`LaTeX rendering failed for: ${expr.content}`, error);
        elements.push(
          <span key={elementKey++} className="text-red-600 font-mono text-sm">
            ${expr.content}$
          </span>
        );
      }
      
      currentIndex = expr.end;
    });
    
    // Add remaining text
    if (currentIndex < text.length) {
      const remainingText = text.slice(currentIndex);
      if (remainingText) {
        elements.push(<span key={elementKey++}>{remainingText}</span>);
      }
    }
    
    return elements.length > 0 ? elements : [text];
  };
  
  const shouldTruncate = !expanded && abstract.length > maxLength;
  const displayContent = shouldTruncate 
    ? processLatexContent(abstract.slice(0, maxLength) + '...')
    : processedContent;
  
  if (!abstract) {
    return (
      <div className={clsx('text-secondary-500 italic', className)}>
        No abstract available
      </div>
    );
  }
  
  return (
    <div className={clsx('prose prose-sm max-w-none', className)}>
      <div className="mb-2">
        <span className="font-medium text-secondary-900">Abstract: </span>
        {hasLatex && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 ml-2">
            LaTeX
          </span>
        )}
      </div>
      
      <div className="text-secondary-700 leading-relaxed">
        {displayContent}
      </div>
      
      {showToggle && abstract.length > maxLength && onToggle && (
        <button
          onClick={onToggle}
          className="mt-2 flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors"
        >
          {expanded ? (
            <>
              <ChevronUpIcon className="w-4 h-4" />
              Show less
            </>
          ) : (
            <>
              <ChevronDownIcon className="w-4 h-4" />
              Show more
            </>
          )}
        </button>
      )}
      
      {shouldTruncate && (
        <div className="mt-1 text-xs text-secondary-500">
          Abstract truncated ({abstract.length} characters)
        </div>
      )}
    </div>
  );
};
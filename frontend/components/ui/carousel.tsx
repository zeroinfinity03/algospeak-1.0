"use client";

import { IconArrowNarrowRight } from "@tabler/icons-react";
import { useState, useRef, useId, useEffect } from "react";

interface SlideData {
  title: string;
  subtitle?: string;
  content?: string[];
  button: string;
  src: string;
}

interface SlideProps {
  slide: SlideData;
  index: number;
  current: number;
  handleSlideClick: (index: number) => void;
}

const Slide = ({ slide, index, current, handleSlideClick }: SlideProps) => {
  const slideRef = useRef<HTMLLIElement>(null);
  const xRef = useRef(0);
  const yRef = useRef(0);
  const frameRef = useRef<number>();

  useEffect(() => {
    const animate = () => {
      if (!slideRef.current) return;
      const x = xRef.current;
      const y = yRef.current;
      slideRef.current.style.setProperty("--x", `${x}px`);
      slideRef.current.style.setProperty("--y", `${y}px`);
      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  const handleMouseMove = (event: React.MouseEvent) => {
    const el = slideRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    xRef.current = event.clientX - (r.left + Math.floor(r.width / 2));
    yRef.current = event.clientY - (r.top + Math.floor(r.height / 2));
  };

  const handleMouseLeave = () => {
    xRef.current = 0;
    yRef.current = 0;
  };

  const imageLoaded = (event: React.SyntheticEvent<HTMLImageElement>) => {
    event.currentTarget.style.opacity = "1";
  };

  const { src, button, title, subtitle, content } = slide;

  return (
    <div className="[perspective:1200px] [transform-style:preserve-3d]">
      <li
        ref={slideRef}
        className="flex flex-1 flex-col items-center justify-center relative text-center text-white opacity-100 transition-all duration-300 ease-in-out w-[95vw] h-[70vmin] mx-[4vmin] z-10 "
        onClick={() => handleSlideClick(index)}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          transform:
            current !== index
              ? "scale(0.98) rotateX(8deg)"
              : "scale(1) rotateX(0deg)",
          transition: "transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
          transformOrigin: "bottom",
        }}
      >
        <div
          className="absolute top-0 left-0 w-full h-full bg-[#1D1F2F] rounded-[1%] overflow-hidden transition-all duration-150 ease-out"
          style={{
            transform:
              current === index
                ? "translate3d(calc(var(--x) / 30), calc(var(--y) / 30), 0)"
                : "none",
          }}
        >
          {src && src.trim() !== "" ? (
            <>
              <img
                className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] object-cover opacity-100 transition-opacity duration-600 ease-in-out"
                style={{
                  opacity: current === index ? 1 : 0.5,
                }}
                alt={title}
                src={src}
                onLoad={imageLoaded}
                loading="eager"
                decoding="sync"
              />
              {current === index && (
                <div className="absolute inset-0 bg-black/60 transition-all duration-1000" />
              )}
            </>
          ) : (
            // Solid background for slides without images
            <div className="absolute inset-0 bg-gradient-to-br from-slate-800 via-slate-700 to-slate-900" />
          )}
        </div>
        <article
          className={`relative p-[3vmin] max-w-4xl transition-opacity duration-1000 ease-in-out overflow-y-auto max-h-[60vmin] scrollbar-hide ${
            current === index ? "opacity-100 visible" : "opacity-0 invisible"
          } ${
            src && src.trim() !== "" && title.includes("TRAINING RESULTS") ? "ml-[30%] mr-2 max-w-xl bg-black/80 backdrop-blur-sm rounded-lg text-xs" : ""
          }`}
        >
          <h2 className="text-lg md:text-2xl lg:text-3xl font-bold mb-2 relative">
            {title}
          </h2>
          {subtitle && (
            <h3 className="text-sm md:text-lg lg:text-xl font-medium mb-4 text-gray-200 relative">
              {subtitle}
            </h3>
          )}
          {content && (
            <div className="text-left text-xs md:text-sm lg:text-base space-y-1 mb-6 relative">
              {content.map((line, idx) => (
                <div key={idx} className={line === "" ? "h-2" : ""}>
                  {line && (
                    <p className={
                      line.startsWith("•") ? "ml-4" : 
                      line.startsWith("🔍") || line.startsWith("📊") || line.startsWith("🏗️") || line.startsWith("🤖") || line.startsWith("⚙️") || line.startsWith("📈") || line.startsWith("❌") || line.startsWith("✅") || line.startsWith("🏆") || line.startsWith("🚀") || line.startsWith("💼") || line.startsWith("🌍") || line.startsWith("🔮") || line.startsWith("📋") ? "font-semibold text-yellow-300 mt-3" :
                      line.startsWith("1️⃣") ? "font-bold text-cyan-300 mt-2" :
                      line.startsWith("2️⃣") ? "font-bold text-purple-300 mt-2" :
                      line.startsWith("3️⃣") ? "font-bold text-orange-300 mt-2" :
                      line.startsWith("4️⃣") ? "font-bold text-pink-300 mt-2" :
                      ""
                    }>
                      {line}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
          <div className="flex justify-center">
            <button className="px-6 py-3 w-fit mx-auto text-sm text-black bg-white border border-transparent flex justify-center items-center rounded-2xl hover:shadow-lg hover:bg-gray-100 transition duration-200 shadow-[0px_2px_3px_-1px_rgba(0,0,0,0.1),0px_1px_0px_0px_rgba(25,28,33,0.02),0px_0px_0px_1px_rgba(25,28,33,0.08)]">
              {button}
            </button>
          </div>
        </article>
      </li>
    </div>
  );
};

interface CarouselControlProps {
  type: string;
  title: string;
  handleClick: () => void;
}

const CarouselControl = ({
  type,
  title,
  handleClick,
}: CarouselControlProps) => {
  return (
    <button
      className={`w-10 h-10 flex items-center mx-2 justify-center bg-neutral-200 dark:bg-neutral-800 border-3 border-transparent rounded-full focus:border-[#6D64F7] focus:outline-none hover:-translate-y-0.5 active:translate-y-0.5 transition duration-200 ${type === "previous" ? "rotate-180" : ""
        }`}
      title={title}
      onClick={handleClick}
    >
      <IconArrowNarrowRight className="text-neutral-600 dark:text-neutral-200" />
    </button>
  );
};

interface CarouselProps {
  slides: SlideData[];
}

export default function Carousel({ slides }: CarouselProps) {
  const [current, setCurrent] = useState(0);

  const handlePreviousClick = () => {
    const previous = current - 1;
    setCurrent(previous < 0 ? slides.length - 1 : previous);
  };

  const handleNextClick = () => {
    const next = current + 1;
    setCurrent(next === slides.length ? 0 : next);
  };

  const handleSlideClick = (index: number) => {
    if (current !== index) {
      setCurrent(index);
    }
  };

  const id = useId();

  return (
    <div
      className="relative w-[95vw] h-[70vmin] mx-auto left-1/2 transform -translate-x-1/2"
      style={{ width: '95vw' }}
      aria-labelledby={`carousel-heading-${id}`}
    >
      <ul
        className="absolute flex mx-[-4vmin] transition-transform duration-1000 ease-in-out"
        style={{
          transform: `translateX(-${current * (100 / slides.length)}%)`,
        }}
      >
        {slides.map((slide, index) => (
          <Slide
            key={index}
            slide={slide}
            index={index}
            current={current}
            handleSlideClick={handleSlideClick}
          />
        ))}
      </ul>
      <div className="absolute flex justify-center w-full top-[calc(100%+1rem)]">
        <CarouselControl
          type="previous"
          title="Go to previous slide"
          handleClick={handlePreviousClick}
        />
        <CarouselControl
          type="next"
          title="Go to next slide"
          handleClick={handleNextClick}
        />
      </div>
    </div>
  );
}
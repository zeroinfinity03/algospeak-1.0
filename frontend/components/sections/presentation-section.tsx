import CarouselDemo from "@/components/carousel-demo";

export default function PresentationSection() {
  return (
    <section id="presentation" className="min-h-screen bg-gray-50 dark:bg-neutral-900 flex items-center justify-center pt-20">
      <div className="max-w-6xl mx-auto px-4 text-center">
        <h2 className="text-4xl md:text-6xl font-bold text-black dark:text-white mb-6">
          Presentation
        </h2>
        <p className="text-lg md:text-xl text-neutral-600 dark:text-neutral-300 mb-10">
          Explore our AI-powered content moderation solution through interactive slides.
        </p>
        <CarouselDemo />
      </div>
    </section>
  );
}
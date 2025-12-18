import MainNavbar from "@/components/main-navbar";
import HomeSection from "@/components/sections/home-section";
import PresentationSection from "@/components/sections/presentation-section";
import DemoSection from "@/components/sections/demo-section";

export default function Home() {
  return (
    <div className="min-h-screen">
      <MainNavbar />
      <HomeSection />
      <PresentationSection />
      <DemoSection />
    </div>
  );
}

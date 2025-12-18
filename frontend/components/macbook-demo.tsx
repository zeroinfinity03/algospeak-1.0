import React from "react";
import { MacbookScroll } from "@/components/ui/macbook-scroll";

export default function MacbookDemo() {
  return (
    <div className="w-full overflow-hidden bg-white dark:bg-[#0B0B0F]">
      <MacbookScroll
        title={
          <span>
            Experience our AI Content Moderation <br /> in action.
          </span>
        }
        showGradient={false}
      />
    </div>
  );
}
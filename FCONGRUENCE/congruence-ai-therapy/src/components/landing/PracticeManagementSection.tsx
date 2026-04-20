import { motion, useReducedMotion } from "framer-motion";
import Appointments from "@/assets/PICTURES/Appointments.png";
import Billing from "@/assets/PICTURES/Billing.png";
import TeamManagement from "@/assets/PICTURES/TeamManagement.png";

const features = [
  {
    title: "Appointments",
    caption: "Schedule sessions across your entire team.",
    image: Appointments
  },
  {
    title: "Billing",
    caption: "Track invoices, payments, and revenue.",
    image: Billing
  },
  {
    title: "Team Management",
    caption: "Manage clinicians and roles.",
    image: TeamManagement
  }
];

export const PracticeManagementSection = () => {
  const prefersReduced = useReducedMotion();

  const fadeUp = {
    initial: prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: prefersReduced ? { duration: 0 } : { duration: 0.6 },
  };

  return (
    <section className="relative py-20 px-8 md:px-16 overflow-hidden">
      <div className="max-w-[1200px] mx-auto">
        {/* Header */}
        <motion.div {...fadeUp} className="mb-12">
          <h2 className="text-[32px] md:text-[42px] font-medium leading-[1.1] tracking-tight text-foreground mb-6">
            Run your entire practice
          </h2>
          <p className="text-[16px] md:text-[18px] text-muted-foreground font-light max-w-[700px]">
            Scheduling, billing, and team management—all in one platform.
          </p>
        </motion.div>

        {/* Grid */}
        <div className="grid md:grid-cols-3 gap-12">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={prefersReduced ? { duration: 0 } : { duration: 0.6, delay: index * 0.1 }}
              className="group"
            >
              <motion.div
                whileHover={{ y: -8 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                {/* Image */}
                <div className="rounded-2xl overflow-hidden shadow-xl border border-border bg-white group-hover:shadow-2xl transition-shadow duration-300">
                  <img
                    src={feature.image}
                    alt={feature.title}
                    className="w-full h-auto"
                    loading="lazy"
                  />
                </div>

                {/* Text */}
                <div className="space-y-2 px-4">
                  <h3 className="text-[20px] font-medium text-foreground">
                    {feature.title}
                  </h3>
                  <p className="text-[15px] text-muted-foreground font-light">
                    {feature.caption}
                  </p>
                </div>
              </motion.div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

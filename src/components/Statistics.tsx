'use client';

import { motion } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';
import { Database, Cpu, DollarSign, Briefcase } from 'lucide-react';

interface Stat {
    label: string;
    value: number;
    suffix: string;
    prefix?: string;
    icon: React.ElementType;
}

const statistics: Stat[] = [
    {
        label: "Data Pipelines Delivered",
        value: 150,
        suffix: "+",
        icon: Database,
    },
    {
        label: "ML & Agentic Solutions",
        value: 9,
        suffix: "+",
        icon: Cpu,
    },
    {
        label: "Revenue Saved",
        value: 6,
        prefix: "$",
        suffix: "M+",
        icon: DollarSign,
    },
    {
        label: "Years of Experience",
        value: 9,
        suffix: "+",
        icon: Briefcase,
    },
];

const Statistics = () => {
    const [isVisible, setIsVisible] = useState(false);
    const sectionRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                }
            },
            { threshold: 0.1 }
        );

        if (sectionRef.current) {
            observer.observe(sectionRef.current);
        }

        return () => observer.disconnect();
    }, []);

    return (
        <section ref={sectionRef} className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-black to-purple-950/10">
            <div className="max-w-7xl mx-auto">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                    {statistics.map((stat, index) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={isVisible ? { opacity: 1, y: 0 } : {}}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="relative group"
                        >
                            <div className="text-center p-6 bg-white/5 border border-white/10 rounded-xl hover:border-purple-500/50 transition-all hover:-translate-y-2">
                                <div className="flex justify-center mb-4">
                                    <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg group-hover:from-purple-500/30 group-hover:to-pink-500/30 transition-all">
                                        <stat.icon size={32} className="text-purple-400" />
                                    </div>
                                </div>
                                <div className="text-4xl md:text-5xl font-bold text-white mb-2">
                                    {isVisible ? (
                                        <CountUpAnimation
                                            end={stat.value}
                                            duration={2}
                                            prefix={stat.prefix}
                                            suffix={stat.suffix}
                                        />
                                    ) : (
                                        <span>{stat.prefix}{stat.value}{stat.suffix}</span>
                                    )}
                                </div>
                                <div className="text-sm text-gray-400 font-medium">
                                    {stat.label}
                                </div>
                                {/* Glow effect */}
                                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-pink-500/0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

const CountUpAnimation = ({ end, duration, prefix = '', suffix = '' }: { end: number; duration: number; prefix?: string; suffix?: string }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        let startTime: number;
        let animationFrame: number;

        const animate = (timestamp: number) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);

            setCount(Math.floor(progress * end));

            if (progress < 1) {
                animationFrame = requestAnimationFrame(animate);
            }
        };

        animationFrame = requestAnimationFrame(animate);

        return () => cancelAnimationFrame(animationFrame);
    }, [end, duration]);

    return <span>{prefix}{count}{suffix}</span>;
};

export default Statistics;

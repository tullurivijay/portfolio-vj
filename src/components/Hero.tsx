'use client';

import { motion } from 'framer-motion';
import { personalInfo } from '@/data/mockData';
import { ArrowRight } from 'lucide-react';

const Hero = () => {
    return (
        <section id="home" className="min-h-screen flex items-center justify-center relative overflow-hidden pt-16">
            {/* Background Elements */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-600/20 rounded-full blur-3xl" />
            </div>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <h2 className="text-xl md:text-2xl text-purple-400 font-medium mb-4">Hello, I'm</h2>
                    <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
                        {personalInfo.name}
                    </h1>
                    <h3 className="text-2xl md:text-4xl text-gray-400 mb-8">
                        {personalInfo.title}
                    </h3>
                    <p className="max-w-2xl mx-auto text-lg text-gray-300 mb-10 leading-relaxed">
                        {personalInfo.description}
                    </p>

                    <div className="flex items-center justify-center">
                        <a
                            href="#projects"
                            className="px-8 py-3 bg-white text-black rounded-full font-semibold hover:bg-gray-200 transition-colors flex items-center gap-2"
                        >
                            View Work <ArrowRight size={20} />
                        </a>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default Hero;

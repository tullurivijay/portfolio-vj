'use client';

import { architectures, architectureCategories, Architecture } from '@/data/architectureData';
import { useState } from 'react';
import { Layers, Maximize2, X } from 'lucide-react';
import Image from 'next/image';

export default function SystemArchitecture() {
    const [selectedCategory, setSelectedCategory] = useState<string>('All');
    const [expandedDiagram, setExpandedDiagram] = useState<Architecture | null>(null);

    const filteredArchitectures = selectedCategory === 'All'
        ? architectures
        : architectures.filter(arch => arch.category === selectedCategory);

    return (
        <section id="architecture" className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-black to-purple-950/20">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-16">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-900/30 rounded-full border border-purple-500/30 mb-6">
                        <Layers className="w-5 h-5 text-purple-400" />
                        <span className="text-purple-300 font-semibold">System Design</span>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                        System Design HLD
                    </h2>
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                        Deep dive into the technical architecture of enterprise-scale Data Engineering and ML systems
                    </p>
                </div>

                {/* Category Filter */}
                <div className="flex flex-wrap justify-center gap-3 mb-12">
                    {architectureCategories.map((category) => (
                        <button
                            key={category}
                            onClick={() => setSelectedCategory(category)}
                            className={`px-6 py-2 rounded-full font-semibold transition-all ${selectedCategory === category
                                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
                                : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white border border-white/10'
                                }`}
                        >
                            {category}
                        </button>
                    ))}
                </div>

                {/* Architecture Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {filteredArchitectures.map((arch) => (
                        <div
                            key={arch.id}
                            className="group bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/50 transition-all hover:-translate-y-2"
                        >
                            {/* Image Container */}
                            <div className="relative aspect-video bg-black/50 overflow-hidden">
                                <Image
                                    src={arch.imagePath}
                                    alt={arch.title}
                                    fill
                                    className="object-contain p-4 group-hover:scale-105 transition-transform duration-300"
                                />
                                <button
                                    onClick={() => setExpandedDiagram(arch)}
                                    className="absolute top-4 right-4 p-2 bg-black/70 hover:bg-purple-600 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                                    aria-label="Expand diagram"
                                >
                                    <Maximize2 className="w-5 h-5 text-white" />
                                </button>
                                <div className="absolute top-4 left-4 px-3 py-1 bg-purple-600 text-white text-xs font-semibold rounded-full">
                                    {arch.category}
                                </div>
                            </div>

                            {/* Content */}
                            <div className="p-6">
                                <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">
                                    {arch.title}
                                </h3>
                                <p className="text-sm text-purple-300 mb-4 font-semibold">
                                    {arch.projectName}
                                </p>
                                <p className="text-gray-400 mb-6 leading-relaxed">
                                    {arch.description}
                                </p>

                                {/* Highlights */}
                                <div className="mb-6">
                                    <h4 className="text-sm font-semibold text-white mb-3">Key Highlights:</h4>
                                    <ul className="space-y-2">
                                        {arch.highlights.map((highlight, idx) => (
                                            <li key={idx} className="flex items-start gap-2 text-sm text-gray-400">
                                                <span className="text-purple-400 mt-1">▸</span>
                                                <span>{highlight}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Tech Stack */}
                                <div className="flex flex-wrap gap-2">
                                    {arch.techStack.map((tech) => (
                                        <span
                                            key={tech}
                                            className="px-3 py-1 text-xs font-semibold text-purple-300 bg-purple-900/30 rounded-full border border-purple-500/30"
                                        >
                                            {tech}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Expanded Diagram Modal */}
            {expandedDiagram && (
                <div
                    className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
                    onClick={() => setExpandedDiagram(null)}
                >
                    <div className="relative max-w-7xl w-full max-h-[90vh] bg-black/50 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden">
                        <div className="sticky top-0 z-10 flex items-center justify-between p-4 bg-black/80 backdrop-blur-md border-b border-white/10">
                            <h3 className="text-xl font-bold text-white">{expandedDiagram.title}</h3>
                            <button
                                onClick={() => setExpandedDiagram(null)}
                                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                aria-label="Close"
                            >
                                <X className="w-6 h-6 text-white" />
                            </button>
                        </div>
                        <div className="relative w-full h-[calc(90vh-80px)] p-8 overflow-auto">
                            <Image
                                src={expandedDiagram.imagePath}
                                alt={expandedDiagram.title}
                                fill
                                className="object-contain"
                            />
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}

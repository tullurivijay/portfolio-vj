'use client';

import { motion } from 'framer-motion';
import { experience } from '@/data/mockData';
import { Briefcase } from 'lucide-react';

const Experience = () => {
    return (
        <section id="experience" className="py-20 bg-black">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Work Experience</h2>
                    <div className="w-20 h-1 bg-purple-600 mx-auto rounded-full" />
                </motion.div>

                <div className="relative max-w-4xl mx-auto">
                    {/* Vertical Line */}
                    <div className="absolute left-0 md:left-1/2 transform md:-translate-x-1/2 top-0 h-full w-1 bg-white/10" />

                    {experience.map((job, index) => (
                        <motion.div
                            key={job.id}
                            initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.2 }}
                            className={`relative flex flex-col md:flex-row gap-8 mb-12 ${index % 2 === 0 ? 'md:flex-row-reverse' : ''
                                }`}
                        >
                            {/* Timeline Dot */}
                            <div className="absolute left-0 md:left-1/2 transform md:-translate-x-1/2 w-8 h-8 bg-black border-4 border-purple-600 rounded-full z-10 -translate-x-[14px] md:translate-x-[-14px]" />

                            <div className="md:w-1/2 pl-12 md:pl-0">
                                <div className={`bg-white/5 border border-white/10 p-6 rounded-xl hover:border-purple-500/50 transition-colors ${index % 2 === 0 ? 'md:mr-12' : 'md:ml-12'
                                    }`}>
                                    <div className="flex items-center gap-2 mb-2 text-purple-400">
                                        <Briefcase size={18} />
                                        <span className="text-sm font-semibold">{job.period}</span>
                                    </div>
                                    <h3 className="text-xl font-bold text-white mb-1">{job.role}</h3>
                                    <h4 className="text-lg text-gray-400 mb-4">{job.company}</h4>
                                    <p className="text-gray-300 leading-relaxed">{job.description}</p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Experience;

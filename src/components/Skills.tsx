'use client';

import { motion } from 'framer-motion';
import { skills } from '@/data/mockData';

const Skills = () => {
    return (
        <section id="skills" className="py-20 bg-black/50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Skills & Expertise</h2>
                    <div className="w-20 h-1 bg-purple-600 mx-auto rounded-full" />
                </motion.div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    {skills.map((skill, index) => (
                        <motion.div
                            key={skill.name}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="p-6 bg-white/5 border border-white/10 rounded-xl hover:border-purple-500/50 transition-colors group"
                        >
                            <div className="flex flex-col items-center text-center">
                                <div className="p-4 bg-white/5 rounded-full mb-4 group-hover:bg-purple-500/20 transition-colors">
                                    <skill.icon size={32} className="text-purple-400" />
                                </div>
                                <h3 className="text-lg font-semibold text-white mb-2">{skill.name}</h3>
                                <p className="text-sm text-gray-400">{skill.level}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Skills;

'use client';

import { motion } from 'framer-motion';
import { skills } from '@/data/mockData';

const Skills = () => {
    // Group skills by category
    const skillCategories = [
        {
            name: "Data Engineering & Cloud",
            skills: skills.filter(s =>
                s.name.includes('Python') ||
                s.name.includes('Databricks') ||
                s.name.includes('AWS') ||
                s.name.includes('Apache') ||
                s.name.includes('Snowflake')
            )
        },
        {
            name: "AI & Machine Learning",
            skills: skills.filter(s =>
                s.name.includes('LLM') ||
                s.name.includes('RAG') ||
                s.name.includes('BERT') ||
                s.name.includes('Machine Learning') ||
                s.name.includes('Weights')
            )
        },
        {
            name: "DevOps & Infrastructure",
            skills: skills.filter(s =>
                s.name.includes('Docker') ||
                s.name.includes('Kubernetes') ||
                s.name.includes('Terraform') ||
                s.name.includes('Jenkins')
            )
        }
    ];

    return (
        <section id="skills" className="py-20 bg-black/50 relative overflow-hidden">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/4 -left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl animate-pulse" />
                <div className="absolute bottom-1/4 -right-1/4 w-96 h-96 bg-pink-600/10 rounded-full blur-3xl animate-pulse delay-700" />
            </div>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Skills & Expertise</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto mb-6">
                        Specialized in building scalable data infrastructure, ML systems, and cutting-edge AI solutions
                    </p>
                    <div className="w-20 h-1 bg-gradient-to-r from-purple-600 to-pink-600 mx-auto rounded-full" />
                </motion.div>

                {/* Skills by Category */}
                {skillCategories.map((category, categoryIndex) => (
                    <motion.div
                        key={category.name}
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: categoryIndex * 0.2 }}
                        className="mb-16 last:mb-0"
                    >
                        {/* Category Title */}
                        <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600 mb-8 text-center">
                            {category.name}
                        </h3>

                        {/* Skills Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {category.skills.map((skill, index) => (
                                <motion.div
                                    key={skill.name}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.3, delay: index * 0.1 }}
                                    whileHover={{ scale: 1.05, y: -5 }}
                                    className="relative group"
                                >
                                    {/* Card */}
                                    <div className="p-6 bg-gradient-to-br from-white/5 to-white/0 border border-white/10 rounded-xl hover:border-purple-500/50 transition-all backdrop-blur-sm">
                                        {/* Icon */}
                                        <div className="flex items-center gap-4 mb-4">
                                            <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg group-hover:from-purple-500/30 group-hover:to-pink-500/30 transition-all">
                                                <skill.icon size={28} className="text-purple-400" />
                                            </div>
                                            <div className="flex-1">
                                                <h4 className="text-lg font-bold text-white group-hover:text-purple-300 transition-colors">
                                                    {skill.name}
                                                </h4>
                                                <p className="text-sm text-gray-400">{skill.level}</p>
                                            </div>
                                        </div>

                                        {/* Progress Bar */}
                                        <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                whileInView={{
                                                    width: skill.level === 'Expert' ? '95%' : skill.level === 'Advanced' ? '85%' : '75%'
                                                }}
                                                viewport={{ once: true }}
                                                transition={{ duration: 1, delay: index * 0.1 + 0.3 }}
                                                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                                            />
                                        </div>

                                        {/* Hover Glow Effect */}
                                        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-pink-500/0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                ))}
            </div>
        </section>
    );
};

export default Skills;

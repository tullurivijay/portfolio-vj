'use client';

import { motion } from 'framer-motion';
import { blogPosts, blogCategories } from '@/data/blogPosts';
import { Calendar, Clock, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';

const Blog = () => {
    const [selectedCategory, setSelectedCategory] = useState('All');

    const filteredPosts = selectedCategory === 'All'
        ? blogPosts
        : blogPosts.filter(post => post.category === selectedCategory);

    return (
        <section id="blog" className="py-20 bg-black">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Technical Blog</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto mb-8">
                        Deep dives into Data Engineering, ML, AI, and emerging technologies
                    </p>
                    <div className="w-20 h-1 bg-purple-600 mx-auto rounded-full" />
                </motion.div>

                {/* Category Filter */}
                <div className="flex flex-wrap justify-center gap-3 mb-12">
                    {blogCategories.map((category) => (
                        <button
                            key={category}
                            onClick={() => setSelectedCategory(category)}
                            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${selectedCategory === category
                                    ? 'bg-purple-600 text-white'
                                    : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                                }`}
                        >
                            {category}
                        </button>
                    ))}
                </div>

                {/* Blog Posts Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {filteredPosts.map((post, index) => (
                        <motion.article
                            key={post.id}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="group bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/50 transition-all hover:-translate-y-2"
                        >
                            <div className="p-6">
                                {/* Category Badge */}
                                <span className="inline-block px-3 py-1 text-xs font-semibold text-purple-300 bg-purple-900/30 rounded-full border border-purple-500/30 mb-4">
                                    {post.category}
                                </span>

                                {/* Title */}
                                <h3 className="text-xl font-bold text-white mb-3 group-hover:text-purple-400 transition-colors line-clamp-2">
                                    {post.title}
                                </h3>

                                {/* Excerpt */}
                                <p className="text-gray-400 mb-6 line-clamp-3">
                                    {post.excerpt}
                                </p>

                                {/* Meta Info */}
                                <div className="flex items-center gap-4 text-sm text-gray-500 mb-6">
                                    <div className="flex items-center gap-1">
                                        <Calendar size={14} />
                                        <span>{new Date(post.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <Clock size={14} />
                                        <span>{post.readTime}</span>
                                    </div>
                                </div>

                                {/* Tags */}
                                <div className="flex flex-wrap gap-2 mb-6">
                                    {post.tags.slice(0, 3).map((tag) => (
                                        <span
                                            key={tag}
                                            className="px-2 py-1 text-xs text-gray-400 bg-white/5 rounded"
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                </div>

                                {/* Read More Link */}
                                <Link
                                    href={`/blog/${post.slug}`}
                                    className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors font-medium"
                                >
                                    Read Article
                                    <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                                </Link>
                            </div>
                        </motion.article>
                    ))}
                </div>

                {/* View All Link */}
                <div className="text-center mt-12">
                    <Link
                        href="/blog"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors font-semibold"
                    >
                        View All Articles
                        <ArrowRight size={20} />
                    </Link>
                </div>
            </div>
        </section>
    );
};

export default Blog;

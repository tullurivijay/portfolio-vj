import { blogPosts, blogCategories } from '@/data/blogPosts';
import Link from 'next/link';
import { Calendar, Clock, ArrowRight, ArrowLeft } from 'lucide-react';

export const metadata = {
    title: 'Technical Blog | Vijay Tulluri',
    description: 'In-depth articles on Data Engineering, Machine Learning, AI, and emerging technologies',
};

export default function BlogPage() {
    return (
        <div className="min-h-screen bg-black text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors"
                    >
                        <ArrowLeft size={20} />
                        Back to Home
                    </Link>
                </div>
            </header>

            {/* Hero */}
            <section className="py-20 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto text-center">
                    <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
                        Technical Blog
                    </h1>
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
                        Deep dives into Data Engineering, Machine Learning, AI, and emerging technologies.
                        Complex concepts explained simply with real-world examples.
                    </p>
                    <div className="w-20 h-1 bg-gradient-to-r from-purple-600 to-pink-600 mx-auto rounded-full" />
                </div>
            </section>

            {/* Blog Posts */}
            <section className="pb-20 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {blogPosts.map((post) => (
                            <article
                                key={post.id}
                                className="group bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/50 transition-all hover:-translate-y-2"
                            >
                                <div className="p-8">
                                    {/* Category Badge */}
                                    <span className="inline-block px-3 py-1 text-xs font-semibold text-purple-300 bg-purple-900/30 rounded-full border border-purple-500/30 mb-4">
                                        {post.category}
                                    </span>

                                    {/* Title */}
                                    <h2 className="text-2xl font-bold text-white mb-4 group-hover:text-purple-400 transition-colors">
                                        {post.title}
                                    </h2>

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
                                        {post.tags.slice(0, 4).map((tag) => (
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
                                        className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors font-semibold"
                                    >
                                        Read Full Article
                                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                                    </Link>
                                </div>
                            </article>
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
}

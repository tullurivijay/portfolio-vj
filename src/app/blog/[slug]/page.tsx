import { blogPosts } from '@/data/blogPosts';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { Calendar, Clock, ArrowLeft, Tag } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export async function generateStaticParams() {
    return blogPosts.map((post) => ({
        slug: post.slug,
    }));
}

export async function generateMetadata({ params }: { params: { slug: string } }) {
    const post = blogPosts.find(p => p.slug === params.slug);

    if (!post) {
        return {
            title: 'Post Not Found',
        };
    }

    return {
        title: `${post.title} | Vijay Tulluri`,
        description: post.excerpt,
    };
}

export default function BlogPost({ params }: { params: { slug: string } }) {
    const post = blogPosts.find(p => p.slug === params.slug);

    if (!post) {
        notFound();
    }

    return (
        <div className="min-h-screen bg-black text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <Link
                        href="/#blog"
                        className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors"
                    >
                        <ArrowLeft size={20} />
                        Back to Blog
                    </Link>
                </div>
            </header>

            {/* Article */}
            <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                {/* Category Badge */}
                <span className="inline-block px-4 py-2 text-sm font-semibold text-purple-300 bg-purple-900/30 rounded-full border border-purple-500/30 mb-6">
                    {post.category}
                </span>

                {/* Title */}
                <h1 className="text-4xl md:text-5xl font-bold text-white mb-6 leading-tight">
                    {post.title}
                </h1>

                {/* Meta Info */}
                <div className="flex flex-wrap items-center gap-6 text-gray-400 mb-8 pb-8 border-b border-white/10">
                    <div className="flex items-center gap-2">
                        <Calendar size={18} />
                        <span>{new Date(post.date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Clock size={18} />
                        <span>{post.readTime}</span>
                    </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-12">
                    {post.tags.map((tag) => (
                        <span
                            key={tag}
                            className="inline-flex items-center gap-1 px-3 py-1 text-sm text-gray-300 bg-white/5 rounded-full border border-white/10"
                        >
                            <Tag size={14} />
                            {tag}
                        </span>
                    ))}
                </div>

                {/* Content */}
                <div className="prose prose-invert prose-lg max-w-none">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ children }) => <h1 className="text-3xl font-bold text-white mt-12 mb-6">{children}</h1>,
                            h2: ({ children }) => <h2 className="text-2xl font-bold text-white mt-10 mb-4">{children}</h2>,
                            h3: ({ children }) => <h3 className="text-xl font-bold text-purple-300 mt-8 mb-3">{children}</h3>,
                            h4: ({ children }) => <h4 className="text-lg font-semibold text-white mt-6 mb-2">{children}</h4>,
                            p: ({ children }) => <p className="text-gray-300 leading-relaxed mb-6">{children}</p>,
                            ul: ({ children }) => <ul className="list-disc list-inside text-gray-300 mb-6 space-y-2">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal list-inside text-gray-300 mb-6 space-y-2">{children}</ol>,
                            li: ({ children }) => <li className="text-gray-300">{children}</li>,
                            code: ({ className, children }) => {
                                const isBlock = className?.includes('language-');
                                if (isBlock) {
                                    return (
                                        <code className="block bg-white/5 border border-white/10 rounded-lg p-4 text-sm overflow-x-auto mb-6 text-purple-300">
                                            {children}
                                        </code>
                                    );
                                }
                                return <code className="bg-white/10 px-2 py-1 rounded text-purple-300 text-sm">{children}</code>;
                            },
                            pre: ({ children }) => <pre className="bg-white/5 border border-white/10 rounded-lg p-4 overflow-x-auto mb-6">{children}</pre>,
                            blockquote: ({ children }) => (
                                <blockquote className="border-l-4 border-purple-500 pl-4 italic text-gray-400 my-6">
                                    {children}
                                </blockquote>
                            ),
                            a: ({ href, children }) => (
                                <a href={href} className="text-purple-400 hover:text-purple-300 underline" target="_blank" rel="noopener noreferrer">
                                    {children}
                                </a>
                            ),
                            table: ({ children }) => (
                                <div className="overflow-x-auto mb-6">
                                    <table className="min-w-full border border-white/10 rounded-lg">
                                        {children}
                                    </table>
                                </div>
                            ),
                            th: ({ children }) => (
                                <th className="border border-white/10 bg-white/5 px-4 py-2 text-left font-semibold text-white">
                                    {children}
                                </th>
                            ),
                            td: ({ children }) => (
                                <td className="border border-white/10 px-4 py-2 text-gray-300">
                                    {children}
                                </td>
                            ),
                        }}
                    >
                        {post.content}
                    </ReactMarkdown>
                </div>

                {/* Back to Blog */}
                <div className="mt-16 pt-8 border-t border-white/10">
                    <Link
                        href="/#blog"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors font-semibold"
                    >
                        <ArrowLeft size={20} />
                        Back to All Articles
                    </Link>
                </div>
            </article>
        </div>
    );
}

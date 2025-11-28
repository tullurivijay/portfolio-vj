'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';

const companies = [
    { name: 'Nike', logo: '/companies/nike.png' },
    { name: 'Apple', logo: '/companies/apple.png' },
    { name: 'Vanguard', logo: '/companies/vanguard.svg' },
    { name: 'Capital One', logo: '/companies/capital-one.svg' },
    { name: 'University of North Texas', logo: '/companies/unt.svg' },
    { name: 'Invesco', logo: '/companies/invesco.svg' },
];

const Companies = () => {
    return (
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/30">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-12"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                        Trusted by Industry Leaders
                    </h2>
                    <p className="text-xl text-purple-400 font-medium">
                        Delivered Exclusive Growth Hacks & Enterprise Solutions
                    </p>
                    <div className="w-20 h-1 bg-gradient-to-r from-purple-600 to-pink-600 mx-auto rounded-full mt-6" />
                </motion.div>

                <div className="relative w-full overflow-hidden">
                    <div className="flex gap-16 animate-marquee whitespace-nowrap">
                        {[...companies, ...companies].map((company, index) => (
                            <div
                                key={`${company.name}-${index}`}
                                className="relative flex flex-col items-center flex-shrink-0"
                            >
                                <div className="relative w-48 h-32 flex items-center justify-center p-6 bg-white border border-white/10 rounded-xl hover:border-purple-500/50 transition-all hover:-translate-y-1 mb-4 shadow-lg">
                                    <div className="relative w-full h-full transition-all duration-300">
                                        <Image
                                            src={company.logo}
                                            alt={company.name}
                                            fill
                                            className="object-contain"
                                        />
                                    </div>
                                </div>
                                <span className="text-gray-400 text-base font-medium transition-colors text-center">
                                    {company.name}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Companies;

'use client';

import { personalInfo } from '@/data/mockData';
import { Github, Linkedin, Mail, Heart } from 'lucide-react';

const Footer = () => {
    return (
        <footer id="contact" className="bg-black border-t border-white/10 pt-20 pb-10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex flex-col items-center text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">Let's Work Together</h2>
                    <p className="text-gray-400 max-w-xl mb-8">
                        Interested in consulting, collaboration, or have a project in mind? Schedule a consultation or send me a message.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center gap-4 mb-12">
                        <a
                            href="https://calendly.com/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-8 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors"
                        >
                            Schedule Consultation
                        </a>
                        <a
                            href={`mailto:${personalInfo.email}`}
                            className="px-8 py-3 border border-white/20 text-white rounded-full font-semibold hover:bg-white/10 transition-colors flex items-center gap-2"
                        >
                            <Mail size={20} />
                            Send Email
                        </a>
                    </div>

                    <div className="flex items-center gap-6 mb-12">
                        <a
                            href={personalInfo.github}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-3 bg-white/5 rounded-full text-gray-400 hover:text-white hover:bg-white/10 transition-all"
                        >
                            <Github size={24} />
                        </a>
                        <a
                            href={personalInfo.linkedin}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-3 bg-white/5 rounded-full text-gray-400 hover:text-white hover:bg-white/10 transition-all"
                        >
                            <Linkedin size={24} />
                        </a>
                    </div>
                </div>

                <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row items-center justify-center text-sm text-gray-500">
                    <p>© {new Date().getFullYear()} Vijay Tulluri. All rights reserved.</p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;

import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import Statistics from '@/components/Statistics';
import Skills from '@/components/Skills';
import Experience from '@/components/Experience';
import Companies from '@/components/Companies';
import Projects from '@/components/Projects';
import SystemArchitecture from '@/components/SystemArchitecture';
import Blog from '@/components/Blog';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="bg-black min-h-screen text-white selection:bg-purple-500/30">
      <Navbar />
      <Hero />
      <Statistics />
      <Skills />
      <Experience />
      <Companies />
      <Projects />
      <SystemArchitecture />
      <Blog />
      <Footer />
    </main>
  );
}

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function AboutPage() {
    return (
        <div className="flex min-h-screen flex-col">
        <main className="flex-1">
            <section className="py-16">
            <div className="container">
                <Card>
                <div className="flex flex-col gap-4 p-6">
                    <h1 className="text-2xl font-bold">회사소개</h1>
                </div>
                </Card>
            </div>
            </section>
        </main>
        </div>
    );
}

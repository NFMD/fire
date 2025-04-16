"use client";
import type { InferGetStaticPropsType } from 'next';
import Image from "next/image";
import Link from "next/link";
import { Mail, MapPin, Phone, Search, Calendar, User, ChevronRight, Trash } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { read_file, natural_language_write_file } from 'default_api';
import { useEffect, useState } from 'react';

interface Notice {
    id: number;
    title: string;
    author: string;
    date: string;
    views: number;
}

export default function NoticePage() {
  const [notices, setNotices] = useState<Notice[]>([]);
    useEffect(() => {
        const fetchData = async () => {
            const noticesData = await read_file({path:'notices.json'});
            setNotices(noticesData);
        }
        fetchData();
    },[]);
    const handleDelete = async (id: number) => {
        const newNotices = notices.filter((notice) => notice.id !== id);
        await natural_language_write_file({
            path: 'notices.json',
            prompt: JSON.stringify(newNotices),
            language:'json'
        });
        setNotices(newNotices);
    }
  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-1">
        <section className="py-16">
          <div className="container">
            <div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">총 {notices.length}건</span>
                <span className="text-gray-400">|</span>
                <span className="text-gray-600">1 페이지</span>
              </div>
            </div>

            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                  </thead>
                  <tbody>
                    {notices.map((notice:Notice) => (
                      <tr key={notice.id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">{notice.id}</td>
                        <td className="px-4 py-4">
                          <Link
                            href={`/notice/${notice.id}`}
                            className="text-navy-700 hover:text-navy-500 hover:underline font-medium"
                          >
                            {notice.title}
                          </Link>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">
                          <div className="flex items-center">
                            <User className="mr-2 h-4 w-4 text-gray-500" />
                            {notice.author}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">
                          <div className="flex items-center">
                            <Calendar className="mr-2 h-4 w-4 text-gray-500" />
                            {notice.date}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">{notice.views}</td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">
                            <Button onClick={() => handleDelete(notice.id)} variant="ghost" size="icon"><Trash/></Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
}

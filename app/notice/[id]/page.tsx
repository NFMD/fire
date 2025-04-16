"use client";
import Image from "next/image";
import Link from "next/link";
import { Mail, MapPin, Phone, Calendar, User, Eye, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { read_file, natural_language_write_file } from 'default_api';
import { useEffect, useState } from 'react';

export default function NoticeDetailPage({ params }: { params: { id: string } }) {
  const noticeId = Number.parseInt(params.id);
    const [notice, setNotice] = useState(null);
    const [notices, setNotices] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    useEffect(() => {
        const fetchData = async () => {
            const noticesData = await read_file({path:'notices.json'});
            setNotices(noticesData);
            const selectedNotice = noticesData.find((item) => item.id === noticeId);
            setNotice(selectedNotice);
            const selectedIndex = noticesData.findIndex((item) => item.id === noticeId);
            setCurrentIndex(selectedIndex);
            if (selectedNotice){
                selectedNotice.views += 1;
                await natural_language_write_file({
                  path: 'notices.json',
                  prompt: JSON.stringify(noticesData),
                  language: 'json'
                });
            }
        }
        fetchData();
    },[noticeId]);
    if (!notice){
        return (
          <div className="flex min-h-screen flex-col items-center justify-center">
            <h1 className="text-2xl font-bold text-navy-700">존재하지 않는 공지사항입니다.</h1>
            <Link href="/notice" className="mt-4 text-navy-600 hover:underline">
              공지사항 목록으로 돌아가기
            </Link>
          </div>
        );
    }
    // 이전 글, 다음 글 찾기
    const prevNotice = currentIndex < notices.length - 1 ? notices[currentIndex + 1] : null;
    const nextNotice = currentIndex > 0 ? notices[currentIndex - 1] : null;
  return (
    <div className="flex min-h-screen flex-col">
      {/* Header Section */}
        {/* ... (헤더 코드) */}

      <main className="flex-1 py-16">
        <div className="container">
          {/* ... (뒤로가기 코드) */}
          <Card className="overflow-hidden">
            <div className="border-b p-6">
              <h1 className="text-2xl font-bold text-navy-700">{notice.title}</h1>
              <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center">
                  <User className="mr-2 h-4 w-4 text-gray-500" />
                  <span>{notice.author}</span>
                </div>
                <div className="flex items-center">
                  <Calendar className="mr-2 h-4 w-4 text-gray-500" />
                  <span>{notice.date}</span>
                </div>
                <div className="flex items-center">
                  <Eye className="mr-2 h-4 w-4 text-gray-500" />
                  <span>조회 {notice.views}</span>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="prose max-w-none">
                {notice.content.split("\n\n").map((paragraph, index) => (
                  <p key={index} className="mb-4 whitespace-pre-line">
                    {paragraph}
                  </p>
                ))}
              </div>
              {notice.attachments && notice.attachments.length > 0 && (
                // ... (첨부파일 코드)
              )}
            </div>
          </Card>
          {/* ... (이전글 다음글 코드) */}
          <div className="mt-8 flex justify-center">
            <Link href="/notice">
              <Button className="bg-navy-600 hover:bg-navy-700 text-white">목록으로</Button>
            </Link>
          </div>
        </div>
      </main>
      {/* Footer */}
        {/* ... (푸터 코드) */}
    </div>
  );
}

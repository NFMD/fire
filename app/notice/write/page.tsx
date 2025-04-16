"use client";
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { read_file, natural_language_write_file } from 'default_api';

export default function NoticeWritePage() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const router = useRouter();

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    // notices.json 파일 읽기
    const noticesData = await read_file({path:"notices.json"});

    // 새 게시글 데이터 생성
    const newNotice = {
      id: noticesData.length + 1, // 기존 게시글 갯수 + 1
      title: title,
      content: content,
      author: 'admin',
      date: new Date().toISOString(), // 현재 시간
      views: 0,
    };

    // 새 게시글 데이터를 noticesData에 추가
    noticesData.push(newNotice);

    // notices.json 파일에 새 게시글 데이터 쓰기
    await natural_language_write_file({
        path: 'notices.json',
        prompt: JSON.stringify(noticesData),
        language:'json'
    })

    // /notice로 이동
    router.push('/notice');
  };

  return (
    // ... (폼 코드)
  );
}
